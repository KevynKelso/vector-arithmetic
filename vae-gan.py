import matplotlib as mpl

mpl.use("Agg")  # Disable the need for X window environment
import tensorflow.keras.backend as K
from matplotlib import pyplot
from numpy import ones, zeros
from numpy.random import randint
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     Lambda, LeakyReLU, Reshape)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from gan import (define_discriminator, define_gan, define_generator,
                 generate_real_samples, load_real_samples)

# def vae():
# latent_dim = 128  # Number of latent dimension parameters
# input_img = Input(shape=(80, 80, 3))

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(input_img)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# # shape_before_flattening = K.int_shape(x)
# x = Flatten()(x)

# z_mu = Dense(latent_dim, name="z_mu")(x)
# z_log_sigma = Dense(
# latent_dim,
# kernel_initializer="zeros",
# bias_initializer="zeros",
# name="z_log_sigma",
# )(x)

# # sampling function
# def sampling(args):
# z_mu, z_log_sigma = args
# epsilon = K.random_normal(
# shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0
# )
# return z_mu + K.exp(z_log_sigma) * epsilon

# # sample vector from the latent distribution
# z = Lambda(sampling)([z_mu, z_log_sigma])

# # encoder = Model(input_img, z)
# # encoder.summary()
# # decoder takes the latent distribution sample as input
# decoder_input = Input(K.int_shape(z)[1:])
# x = Dense(
# 3200, activation="relu", name="intermediate_decoder", input_shape=(latent_dim,)
# )(decoder_input)
# x = Reshape((5, 5, 128))(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(3, (3, 3), strides=2, padding="same", activation="sigmoid")(x)

# # decoder model statement
# decoder = Model(decoder_input, x)

# # apply the decoder to the sample from the latent distribution
# pred = decoder(z)

# def vae_loss(x, pred):
# x = K.flatten(x)
# pred = K.flatten(pred)
# # Reconstruction loss
# reconst_loss = 1000 * K.mean(K.square(x - pred))

# # KL divergence
# kl_loss = -0.5 * K.mean(
# 1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1
# )

# return reconst_loss + kl_loss

# # VAE model statement
# vae = Model(input_img, pred, name="VAE")
# vae.add_loss(vae_loss(input_img, pred))
# # optimizer = Adam(learning_rate=0.0005)
# # vae.compile(optimizer=optimizer, loss=None)

# # vae.summary()
# return vae
# AE VERSION 1
# def ae():
# latent_dim = 128  # Number of latent dimension parameters
# input_img = Input(shape=(80, 80, 3))

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(input_img)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
# x = BatchNormalization()(x)

# # shape_before_flattening = K.int_shape(x)
# x = Flatten()(x)

# x = Dense(latent_dim, activation="relu")(x)

# x = Dense(
# 3200, activation="relu", name="intermediate_decoder", input_shape=(latent_dim,)
# )(x)
# x = Reshape((5, 5, 128))(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
# x = BatchNormalization()(x)

# x = Conv2DTranspose(3, (3, 3), strides=2, padding="same", activation="sigmoid")(x)

# # AE model statement
# ae = Model(input_img, x, name="AE")

# return ae
# AE VERSION 2, based almost entirely on discriminator / generator
def ae(in_shape=(80, 80, 3)):
    latent_dim = 100
    model = Sequential(name="Encoder")
    # normal
    model.add(Conv2D(128, (5, 5), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dense(latent_dim, activation="relu"))
    model.add(define_generator(latent_dim))

    return ae


def save_plot(examples, epoch, n=10):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = "generated_plot_e%03d.png" % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples, dataset)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = "generator_model_ae2_%03d.h5" % (epoch + 1)
    g_model.save(filename)
    filename = "discriminator_model_ae2_%03d.h5" % (epoch + 1)
    d_model.save(filename)


def generate_fake_samples(vae_model, latent_dim, n_samples, dataset):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x_input = dataset[ix]
    # use VAE to reconstruct some dataset images
    X = vae_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))

    return X, y


def train(ae_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            # updae AE model weights
            # ae_loss1, _ = ae_model.train_on_batch(X_real, X_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(
                ae_model, latent_dim, half_batch, dataset
            )
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            # X_gan = generate_latent_points(latent_dim, n_batch)
            ix = randint(0, dataset.shape[0], n_batch)
            X_gan = dataset[ix]
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print(
                f">{i+1}, {j+1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}"
            )
        # evaluate the model performance, sometimes
        # if (i + 1) % 10 == 0:
        summarize_performance(i, ae_model, d_model, dataset, latent_dim)


def main():
    dataset = load_real_samples()
    d_model = define_discriminator()
    ae_model = ae()

    gan_model = define_gan(ae_model, d_model)

    train(ae_model, d_model, gan_model, dataset, 128)


if __name__ == "__main__":
    main()
