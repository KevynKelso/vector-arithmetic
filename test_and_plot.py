import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

from vae_gan import load_real_samples, save_plot

# from pyfzf.pyfzf import FzfPrompt

# fzf = FzfPrompt()

MODEL_NAME = "aegan4"


def test_model():
    epoch = 23
    dataset = load_real_samples()
    save_plot(dataset, 0, n=3, filename="dataset_orig.png", show=True)
    generator_model = load_model(f"generator_model_{MODEL_NAME}_{epoch}.h5")
    y = generator_model(dataset)
    save_plot(y, 0, n=3, filename="e23_output.png", show=True)


def plot_losses():
    df = pd.read_csv(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv")
    plt.title("Loss while training: alpha = 1, beta = 0")
    plt.xlabel("1/2 batch")
    plt.ylabel("loss")
    plt.plot(df["ae_loss"], label="Reconstruction Loss")
    plt.plot(df["gan_loss"] * 0.005, label="GAN loss")
    plt.legend()
    plt.show()


def plot_discriminator_accuracy():
    df = pd.read_csv(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv")
    plt.title("Discriminator Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.plot(df["acc_real"] * 100, label="Accuracy real")
    plt.plot(df["acc_fake"] * 100, label="Accuracy fake")
    plt.legend()
    plt.show()


def main():
    plot_losses()
    plot_discriminator_accuracy()


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot
