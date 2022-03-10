import threading
from tkinter import BOTTOM, CENTER, Button, DoubleVar, Label, Scale, Tk

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

IMG_WIDTH = 80
IMG_HEIGHT = 80
updating = False

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


def get_tk_imgs(g_model, latent_dim):
    latent = generate_latent_points(latent_dim, 1)
    X = g_model.predict(latent)[0]
    X = ((X + 1) / 2.0 * 255 / np.max(X)).astype(np.uint8)
    img_pil = Image.fromarray(X)
    tk_img_output = ImageTk.PhotoImage(img_pil)

    return tk_img_output, latent[0]


def set_scales_to_latent_vec(scales, latent_vec):
    latent_arr = np.array(latent_vec)
    for scale, latent_val in zip(scales, latent_arr):
        scale.set(latent_val)


def update_plot_thread(latent, plot_image):
    global updating
    if updating:
        return

    updating = True

    plt.clf()
    plt.hist(latent)
    plt.savefig("test.png")
    plot = ImageTk.PhotoImage(Image.open("test.png"))
    plot_image.configure(image=plot)
    plot_image.image = plot
    updating = False


def gui(g_model, latent_dim):
    ws = Tk()
    ws.title("Vector Arithmetic GUI")
    scales = []
    scale_values = []

    def change_img(g_model, output_img_label, latent_dim, plot_image):
        output_img, latent = get_tk_imgs(g_model, latent_dim)

        output_img_label.configure(image=output_img)
        output_img_label.image = output_img

        set_scales_to_latent_vec(scale_values, latent)
        update_plot_thread(latent, plot_image)

    def decode_new_latent(output_img, plot_image):
        new_latent = np.asarray([scale.get() for scale in scales]).reshape(
            1, 100
        )  # scales are already 'normalized'
        x = threading.Thread(
            target=update_plot_thread,
            args=(
                new_latent[0],
                plot_image,
            ),
        )
        x.start()

        X = g_model.predict(new_latent)[0]
        X = ((X + 1) / 2.0 * 255 / np.max(X)).astype(np.uint8)
        output_pil = Image.fromarray(X)

        tk_img = ImageTk.PhotoImage(output_pil)

        output_img.configure(image=tk_img)
        output_img.image = tk_img

    img_default_output, latent = get_tk_imgs(g_model, latent_dim)

    output_image = Label(
        ws,
        justify=CENTER,
        compound=BOTTOM,
        pady=5,
        text="Output image GAN",
        image=img_default_output,
    )
    plot_image = Label(
        ws,
        justify=CENTER,
        compound=BOTTOM,
        pady=5,
        text="Plot",
    )
    update_plot_thread(latent, plot_image)

    scale_values = [DoubleVar() for _ in range(latent_dim)]
    scales = [
        Scale(
            ws,
            from_=-5,
            to=5,
            resolution=1e-10,
            showvalue=0,
            command=lambda _: decode_new_latent(output_image, plot_image),
            variable=scale_values[i],
        )
        for i in range(latent_dim)
    ]

    set_scales_to_latent_vec(scale_values, latent)

    button = Button(
        ws,
        text="Generate new image",
        command=lambda: change_img(g_model, output_image, latent_dim, plot_image),
    )

    output_image.grid(row=0, column=0)
    button.grid(row=1, column=0)
    plot_image.grid(row=2, column=0)

    num_cols = 50
    for i, scale in enumerate(scales):
        scale.grid(row=(int(i / num_cols) % num_cols), column=((i + 1) % num_cols) + 1)

    ws.mainloop()


def main():
    model = load_model("./rsync-receiver/generator_model_040.h5")
    gui(model, 100)


if __name__ == "__main__":
    main()
