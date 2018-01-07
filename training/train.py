import argparse

from autoencodermodels import basic_convolutional_autoencoder
from imagestoarray import generate_array

PATH = 'pitbull/images/'

def train_model(model_name='model', epochs=10):
    autoencoder = basic_convolutional_autoencoder(input_shape=(None, None, 3))
    autoencoder.fit(x_train, x_train,
                    epochs=int(epochs),
                    batch_size=1,
                    shuffle=True)
    autoencoder.save(model_name + '.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--epochs', required=True)

    x_train = generate_array(PATH)

    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
