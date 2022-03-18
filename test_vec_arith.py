import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model, load_model

from data_preprocessing import plot_faces
from gui_ml import get_vae
from vae import load_real_samples


def main():
    dataset = load_real_samples()
    encoder, decoder = get_vae("./models/conv-vae-1.h5")

    smiling_women = np.asarray([dataset[3], dataset[5], dataset[9]])
    neutral_women = np.asarray([dataset[0], dataset[4], dataset[10]])
    neutral_men = np.asarray([dataset[6], dataset[8], dataset[28]])

    avg_smiling_women_lv = np.mean(encoder.predict(smiling_women), axis=0)
    avg_neutral_women_lv = np.mean(encoder.predict(neutral_women), axis=0)
    avg_neutral_men_lv = np.mean(encoder.predict(neutral_men), axis=0)

    # avg_smiling_women = decoder.predict(np.asarray([avg_smiling_women_lv]))[0]
    # avg_neutral_women = decoder.predict(np.asarray([avg_neutral_women_lv]))[0]
    # avg_neutral_men = decoder.predict(np.asarray([avg_neutral_men_lv]))[0]

    # plt.imshow((avg_smiling_women + 1) / 2.0)
    # plt.show()
    # plt.imshow((avg_neutral_women + 1) / 2.0)
    # plt.show()
    # plt.imshow((avg_neutral_men + 1) / 2.0)
    # plt.show()

    combined_vector = avg_smiling_women_lv - avg_neutral_women_lv * avg_neutral_men_lv

    output = decoder.predict(np.asarray([combined_vector]))
    plot_faces((output + 1) / 2.0, 1)


def plot_latent_space():
    dataset = load_real_samples()
    encoder, _ = get_vae("./models/conv-vae-1.h5")
    latent_vecs = encoder.predict(dataset)

    y0 = []
    y1 = []
    y2 = []
    for vector in latent_vecs:
        y0.append(vector[0])
        y1.append(vector[1])
        y2.append(vector[2])

    plt.suptitle("Distributions for 3 values of the latent vector over 90 predictions")
    plt.subplot(1, 3, 1)
    plt.hist(y0)
    plt.title("latent_space[0]")
    plt.subplot(1, 3, 2)
    plt.hist(y1)
    plt.title("latent_space[1]")
    plt.subplot(1, 3, 3)
    plt.hist(y2)
    plt.title("latent_space[2]")
    plt.show()


if __name__ == "__main__":
    plt.axis("off")
    main()
    # plot_latent_space()
# plot_faces((np.asarray([avg_smiling_women]) + 1) / 2.0, 1)
# plot_faces((np.asarray([avg_neutral_women]) + 1) / 2.0, 1)
# plot_faces((np.asarray([avg_neutral_men]) + 1) / 2.0, 1)

# arrays = np.vstack((smiling_women, neutral_women, neutral_men))
# print(arrays.shape)

# for i in range(arrays.shape[0]):
# plot_faces((np.asarray([arrays[i]]) + 1) / 2.0, 1)

# latent = encoder.predict(dataset)
# output = decoder.predict(latent)
# plot_faces((output + 1) / 2.0, 5)
# plt.suptitle("Latent space distributions for image averages")
# plt.subplot(1, 3, 1)
# plt.hist(lv1)
# plt.title("Smiling women")
# plt.subplot(1, 3, 2)
# plt.hist(lv2)
# plt.title("Neutral women")
# plt.subplot(1, 3, 3)
# plt.hist(lv3)
# plt.title("Neutral men")

# plt.show()

# plt.imshow(lv1.reshape((-1, 8)))
# plt.show()
# plt.imshow(lv2.reshape((-1, 8)))
# plt.show()
# plt.imshow(lv3.reshape((-1, 8)))
# plt.show()
# lv1o = decoder.predict(np.asarray([lv1]))[0]
# lv2o = decoder.predict(np.asarray([lv2]))[0]
# lv3o = decoder.predict(np.asarray([lv3]))[0]
# plt.imshow((lv2o + 1) / 2.0)
# plt.show()
# plt.imshow((lv3o + 1) / 2.0)
# plt.show()
# avg_smiling_women = np.average(smiling_women, axis=0)
# avg_neutral_women = np.average(neutral_women, axis=0)
# avg_neutral_men = np.average(neutral_men, axis=0)

# lv1 = encoder.predict(np.asarray([avg_smiling_women]))[0]
# lv2 = encoder.predict(np.asarray([avg_neutral_women]))[0]
# lv3 = encoder.predict(np.asarray([avg_neutral_men]))[0]
