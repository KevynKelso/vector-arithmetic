import matplotlib as mpl
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

mpl.use("Agg")  # Disable the need for X window environment
from matplotlib import pyplot
from numpy import ones, zeros
from numpy.random import randint
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     Lambda, LeakyReLU, Reshape)
from tensorflow.keras.models import Sequential

from gan import (define_discriminator, define_generator,
                 generate_real_samples, load_real_samples)


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

    return model


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


def summarize_performance(epoch, g_model, d_model, dataset, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, n_samples, dataset)
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


def generate_fake_samples(vae_model, n_samples, dataset):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x_input = dataset[ix]
    # use VAE to reconstruct some dataset images
    X = vae_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))

    return X, y


# def train(ae_model, d_model, gan_model, dataset, n_epochs=100, n_batch=128):
def train(ae_model, d_model, gan_model, dataset, n_epochs=100, n_batch=13):
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
            X_fake, y_fake = generate_fake_samples(ae_model, half_batch, dataset)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
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
        summarize_performance(i, ae_model, d_model, dataset)

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    # model = Sequential()
    model = VAEGAN(g_model)
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(my_loss=loss_wapper(g_model, 1, 1), optimizer=opt)

    return model


def loss_wapper(g_model, alpha, beta):
    mse = MeanSquaredError()
    bce = BinaryCrossentropy(from_logits=True)

    def loss(x, y_true, y_pred):
        y = g_model(x)
        ae_loss = alpha * mse(x, y)
        gan_loss = beta * bce(y_true, y_pred)
        print(f"ae_loss = {ae_loss.numpy()} gan_loss = {gan_loss}")
        return ae_loss + gan_loss

    return loss

class VAEGAN(tf.keras.Sequential):
    def compile(self, optimizer, my_loss):
        super().compile(optimizer)
        self.my_loss = my_loss

    def train_step(self, data):
        # data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_value = self.my_loss(x, y, y_pred)

        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss_value": loss_value}

def main():
    dataset = load_real_samples()
    d_model = define_discriminator()
    ae_model = ae()  # AE model is generator

    gan_model = define_gan(ae_model, d_model)

    train(ae_model, d_model, gan_model, dataset)


if __name__ == "__main__":
    main()
