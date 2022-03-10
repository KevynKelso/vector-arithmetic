import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from gan import generate_latent_points, load_real_samples


def main():
    dataset = load_real_samples()
    generator = load_model("./rsync-receiver/generator_model_040.h5")

    latent_vecs = generate_latent_points(100, 5000)
    generated_imgs = generator.predict(latent_vecs)

    min_mse = 1
    best_img = generated_imgs[0]
    for gen_img in generated_imgs:
        mse = ((dataset[0] - generated_imgs[0]) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
            best_img = gen_img

    print(min_mse)
    plt.imshow((dataset[0] + 1) / 2.0)
    plt.show()
    plt.imshow((best_img + 1) / 2.0)
    plt.show()


if __name__ == "__main__":
    main()
