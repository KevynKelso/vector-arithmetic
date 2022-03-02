import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     Lambda, Reshape)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

BUFFER_SIZE = 10000
BATCH_SIZE = 512


def load_real_samples():
    data = np.load("img_align_celeba.npz")
    X = data["arr_0"]
    X = X.astype("float32")
    X = (X - 127.5) / 127.5

    return X


def vae(model_name: str):
    latent_dim = 256

    input_img = Input(shape=(80, 80, 3))
    x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(input_img)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)  # 20 x 20
    x = Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(x)  # 10 x 10
    x = Conv2D(16, (3, 3), activation="relu", padding="same", strides=2)(
        x
    )  # 5 x 5 might want to delete this layer
    x = BatchNormalization()(x)
    x = Flatten()(x)

    z_mu = Dense(latent_dim)(x)
    z_log_sigma = Dense(
        latent_dim, kernel_initializer="zeros", bias_initializer="zeros"
    )(x)
    # sampling function
    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0
        )
        return z_mu + K.exp(z_log_sigma) * epsilon

    # sample vector from the latent distribution
    z = Lambda(sampling)([z_mu, z_log_sigma])
    encoder = Model(input_img, z)

    decoder_input = Input(K.int_shape(z)[1:])
    x = Dense(
        400, activation="relu", name="intermediate_decoder", input_shape=(latent_dim,)
    )(decoder_input)
    x = Reshape((5, 5, 16))(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
    x = Conv2DTranspose(3, (3, 3), strides=2, padding="same")(x)  # 80x80x3

    decoder = Model(decoder_input, x)
    pred = decoder(z)

    def vae_loss(x, pred):
        x = K.flatten(x)
        pred = K.flatten(pred)
        # Reconstruction loss
        reconst_loss = 1000 * K.mean(K.square(x - pred))

        # KL divergence
        kl_loss = -0.5 * K.mean(
            1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1
        )

        return reconst_loss + kl_loss

    # encoder.summary()
    # decoder.summary()
    vae = Model(input_img, pred)
    vae.add_loss(vae_loss(input_img, pred))
    optimizer = Adam(learning_rate=0.0005)
    vae.compile(optimizer=optimizer, loss=None)

    return vae
    # vae.summary()


def main():
    model_name = "conv-vae-1"
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=5, mode="auto"
    )

    dataset_numpy = load_real_samples()
    dataset = tf.data.Dataset.from_tensor_slices(dataset_numpy)
    x_train = dataset.skip(int(dataset_numpy.shape[0] * 0.2))
    x_valid = dataset.take(int(dataset_numpy.shape[0] * 0.2))

    x_train = x_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    x_valid = x_valid.batch(BATCH_SIZE)
    # x_train = dataset[: int(dataset.shape[0] * 0.8), :, :, :]
    # x_valid = dataset[int(dataset.shape[0] * 0.8) :, :, :, :]

    vae_model = vae(model_name)
    history = vae_model.fit(
        x_train,
        epochs=50,
        validation_data=x_valid,
        callbacks=[early_stopping],
        verbose=1,
    )

    with open(f"histories/{model_name}.txt", "w") as f:
        f.write(f"{history}")


if __name__ == "__main__":
    main()
