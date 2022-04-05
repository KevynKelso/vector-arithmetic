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


def main():
    df = pd.read_csv("./alpha_beta_loss_aegan4.csv")
    plt.title("Reconstruction loss")
    plt.xlabel("1/2 batch")
    plt.ylabel("loss")
    plt.plot(df["ae_loss"])
    plt.show()


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot