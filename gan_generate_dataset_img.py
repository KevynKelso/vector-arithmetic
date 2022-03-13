import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from gan import generate_latent_points, load_real_samples


def isReal(label):
    print(label)
    if label > 0.8:
        return True
    return False


def classify_real_fake_imgs(generator, discriminator):
    dataset = load_real_samples()

    labels = discriminator.predict(dataset)

    latent_vecs = generate_latent_points(100, 10000)
    generated_imgs = generator.predict(latent_vecs)
    labels = discriminator.predict(generated_imgs)

    real_imgs = []
    real_latents = []
    fake_imgs = []
    fake_latents = []
    for lv, img, label in zip(latent_vecs, generated_imgs, labels):
        # Are real images better quality that fake images?
        if isReal(label[0]):
            real_imgs.append(img)
            real_latents.append(lv)
            continue
        fake_imgs.append(img)
        fake_latents.append(lv)

    print(f"Real imgs: {len(real_imgs)}")
    print(f"Real latents: {len(real_latents)}")
    print(f"Fake imgs: {len(fake_imgs)}")
    print(f"Fake latents: {len(fake_latents)}")
    np.savez_compressed("real_imgs.npz", real_imgs)
    np.savez_compressed("real_latents.npz", real_latents)
    np.savez_compressed("fake_imgs.npz", fake_imgs)
    np.savez_compressed("fake_latents.npz", fake_latents)


def test_real_fake_imgs():
    real_imgs = np.load("real_imgs.npz")["arr_0"].astype("float32")
    fake_imgs = np.load("fake_imgs.npz")["arr_0"].astype("float32")
    show_plot(real_imgs, "real_imgs.png", 3)
    show_plot(fake_imgs, "fake_imgs.png", 3)


def show_plot(examples, name, n=10):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    plt.savefig(name)


if __name__ == "__main__":
    # generator = load_model("./rsync-receiver/generator_model_040.h5")
    # discriminator = load_model("./models/discriminator_model_020.h5")
    # classify_real_fake_imgs(generator, discriminator)
    test_real_fake_imgs()
