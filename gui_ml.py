import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model

IMG_WIDTH = 80
IMG_HEIGHT = 80


def get_ae(model_name):
    autoencoder = load_model(model_name)

    return autoencoder


def get_vae(model_name):
    variational_autoencoder = load_model(model_name)
    z = variational_autoencoder.layers[9]

    encoder = Model(variational_autoencoder.input, z.output)
    decoder = variational_autoencoder.layers[10]

    return encoder, decoder


def encode_decode(encoder, decoder, img):
    img_arr = np.array(img) / 255
    batch = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3))
    batch[0, :, :, :] = img_arr

    # actual ML
    latent = encoder(batch)
    output = decoder.predict(latent)

    return latent[0], (output * 255).astype(np.uint8)[0]


def run_image_ae(ae, img):
    input_img_arr = np.array(img) / 255
    batch = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3))
    batch[0, :, :, :] = input_img_arr

    output = ae.predict(batch)[0]

    return Image.fromarray((output * 255).astype(np.uint8))


def run_input_output_img_ae(ae, file):
    input_img = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
    output_img = run_image_ae(ae, input_img)

    return input_img, output_img


def run_input_output_imgs(encoder, decoder, file):
    input_img = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
    latent, output_img = encode_decode(encoder, decoder, input_img)
    output_img = Image.fromarray(output_img)

    return input_img, output_img, latent
