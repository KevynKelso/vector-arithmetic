import glob
import random
from tkinter import BOTTOM, CENTER, Button, Label, Scale, Tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from blur_detection import get_blur_factor
from gui_ml import (encode_decode, get_ae, get_vae, run_image_ae,
                    run_input_output_img_ae, run_input_output_imgs)
from vae import load_real_samples

g_latent = []

IMG_WIDTH = 80
IMG_HEIGHT = 80


def get_tk_imgs(encoder, decoder, img_np):
    img_pil, output_img, latent = run_input_output_imgs(encoder, decoder, img_np)

    tk_img_input = ImageTk.PhotoImage(img_pil)
    tk_img_output = ImageTk.PhotoImage(output_img)

    return tk_img_input, tk_img_output, latent


def set_scales_to_latent_vec(scales, latent_vec):
    latent_arr = np.array(latent_vec)
    for scale, latent_val in zip(scales, latent_arr):
        scale.set(latent_val)


def gui(dataset_np, encoder, decoder):
    ws = Tk()
    ws.title("Vector Arithmetic GUI")

    def change_img(input_img_label, output_img_label, scales, length):
        global g_latent

        img_np = dataset_np[random.randint(0, length - 1)]
        input_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        input_blur = get_blur_factor(input_img_pil)

        input_img, output_img, latent = get_tk_imgs(encoder, decoder, img_np)
        g_latent = latent

        text = f"Input, blur:{input_blur}"
        input_img_label.configure(image=input_img, text=text)
        output_img_label.configure(image=output_img)
        input_img_label.image = input_img  # prevents garbage collection
        input_img_label.text = text
        output_img_label.image = output_img

        set_scales_to_latent_vec(scales, g_latent)

    def decode_new_latent(output_img, scales):
        # need to build latent vector from scale values
        new_latent = [
            scale.get() for scale in scales
        ]  # scales are already 'normalized'

        output = decoder.predict([new_latent])[0]

        output_pil = Image.fromarray((output * 255).astype(np.uint8))
        output_blur = get_blur_factor(output_pil)

        tk_img_output = ImageTk.PhotoImage(
            Image.fromarray((output * 255).astype(np.uint8))
        )

        text1 = f"Output, blur:{output_blur}"
        output_img.configure(image=tk_img_output, text=text1)
        output_img.image = tk_img_output
        output_img.text = text1

    (
        img_default_input,
        img_default_output,
        g_latent,
    ) = get_tk_imgs(encoder, decoder, dataset_np[0])

    input_image_vae = Label(
        ws,
        justify=CENTER,
        compound=BOTTOM,
        pady=5,
        text="Input image VAE",
        image=img_default_input,
    )
    output_image_vae = Label(
        ws,
        justify=CENTER,
        compound=BOTTOM,
        pady=5,
        text="Output image VAE",
        image=img_default_output,
    )
    scales = [
        Scale(
            ws,
            from_=-5,
            to=5,
            resolution=1e-10,
            showvalue=0,
            command=lambda _: decode_new_latent(output_image_vae, scales),
        )
        for _ in range(g_latent.shape[0])
    ]

    set_scales_to_latent_vec(scales, g_latent)

    button = Button(
        ws,
        text="Change source image",
        command=lambda: change_img(
            input_image_vae, output_image_vae, scales, dataset_np.shape[0]
        ),
    )
    # button2 = Button(
    # ws,
    # text="Reset sliders",
    # command=lambda: set_scales_to_latent_vec(scales, g_latent),
    # )

    input_image_vae.grid(row=0, column=0)
    button.grid(row=1, column=0)
    output_image_vae.grid(row=2, column=0)
    # button2.grid(row=3, column=0)

    num_cols = 50
    for i, scale in enumerate(scales):
        scale.grid(row=(int(i / num_cols) % num_cols), column=((i + 1) % num_cols) + 1)

    ws.mainloop()


def main():
    dataset_np = load_real_samples()
    encoder, decoder = get_vae("./models/conv-vae-1.h5")
    gui(dataset_np, encoder, decoder)


if __name__ == "__main__":
    main()
