from keras import backend as K
from keras.models import load_model
import os
import numpy as np
import scipy

from imagestoarray import preprocess_image, deprocess_image

MODEL = load_model('model3.h5')

DREAM = MODEL.input

HYPERPARAMETERS = {
    'step':0.01,  # Gradient ascent step size
    'num_octave':3,  # Number of scales at which to run gradient ascent
    'octave_scale':1.4,  # Size ratio between scales
    'iterations':20 , # Number of ascent steps per scale
    'max_loss':10.
}

IMAGE_PATH = 'pitbull/images/' + os.listdir('pitbull/images/')[0]

SETTINGS = None

def define_loss(K=K, settings=SETTINGS, model=MODEL):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    loss = K.variable(0.)
    for layer_name in list(layer_dict.keys()):
        # Add the L2 norm of the features of a layer to the loss.
        # Ignoring settings for now
        #assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        #coeff = settings['features'][layer_name]
        coeff = 1
        x = layer_dict[layer_name].output
        # We avoid border artefacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
        return loss


def compute_gradients(loss, dream=DREAM, K=K):
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    return grads


def eval_loss_and_grads(x):
    #fetch_loss_and_grads defined in at runtime
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def path_to_pyramid(base_image_path=IMAGE_PATH, num_octave=HYPERPARAMETERS['num_octave'],
                    octave_scale=HYPERPARAMETERS['octave_scale']):
    img = preprocess_image(base_image_path)
    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    return successive_shapes, original_img, shrunk_original_img


def deepdream(successive_shapes, img, shrunk_original_img, iterations=HYPERPARAMETERS['iterations'],
              step=HYPERPARAMETERS['step'], max_loss=HYPERPARAMETERS['max_loss']):
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

        return img


if __name__ == '__main__':
    loss = define_loss()
    grads = compute_gradients(loss)
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([DREAM], outputs)
    successive_shapes, original_img, shrunk_original_img = path_to_pyramid()
    print(x.shape for x in successive_shapes)
    dream_img = deepdream(successive_shapes, original_img, shrunk_original_img)
    save_img(dream_img, fname=IMAGE_PATH + 'dream' + '.png')
