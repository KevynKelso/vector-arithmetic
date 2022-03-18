from numpy import load
from tensorflow.keras.models import load_model

from blur_detection import get_average_blur


def main():
    data = load("latent_points.npz")
    points = data["arr_0"]

    baseline_gan = load_model("./rsync-receiver/generator_model_040.h5")
    sm_gan1 = load_model("./small-gan/generator_model_sm1_100.h5")
    sm_gan2 = load_model("./small-gan2/generator_model_sm2_060.h5")

    baseline_gan_predictions = baseline_gan.predict(points)
    sm_gan1_predictions = sm_gan1.predict(points)
    sm_gan2_predictions = sm_gan2.predict(points)

    print(f"baseline blur: {get_average_blur(baseline_gan_predictions)}")
    print(f"sm_gan1 blur: {get_average_blur(sm_gan1_predictions)}")
    print(f"sm_gan2 blur: {get_average_blur(sm_gan2_predictions)}")


if __name__ == "__main__":
    main()
