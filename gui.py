import glob
import random
from tkinter import BOTTOM, CENTER, Button, Label, Scale, Tk

import cv2
import numpy as np
from blur_detection import get_blur_factor
from PIL import Image, ImageTk

from gui_ml import (encode_decode, get_ae, get_vae, run_image_ae,
                    run_input_output_img_ae, run_input_output_imgs)

g_latent = []

IMG_WIDTH = 80
IMG_HEIGHT = 80


def get_tk_imgs(encoder, decoder, ae, file):
    img_pil, output_img, latent = run_input_output_imgs(encoder, decoder, file)
    output_img_ae = run_image_ae(ae, output_img)

    tk_img_input = ImageTk.PhotoImage(img_pil)
    tk_img_output = ImageTk.PhotoImage(output_img)
    tk_img_output_ae = ImageTk.PhotoImage(output_img_ae)

    return tk_img_input, tk_img_output, tk_img_output_ae, latent


def set_scales_to_latent_vec(scales, latent_vec):
    latent_arr = np.array(latent_vec)
    for scale, latent_val in zip(scales, latent_arr):
        scale.set(latent_val)


def gui(files, encoder, decoder, ae):
    ws = Tk()
    ws.title("Latent Space Interpolation GUI")

    def change_img(
        input_img_label, output_img_label, output_img_ae_label, scales, length
    ):
        global g_latent

        file = files[random.randint(0, length - 1)]
        input_img_pil = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
        input_blur = get_blur_factor(input_img_pil)

        input_img, output_img, output_img_ae, latent = get_tk_imgs(
            encoder, decoder, ae, file
        )
        g_latent = latent

        text = f"Input, blur:{input_blur}"
        input_img_label.configure(image=input_img, text=text)
        output_img_label.configure(image=output_img)
        output_img_ae_label.configure(image=output_img_ae)
        input_img_label.image = input_img  # prevents garbage collection
        input_img_label.text = text
        output_img_label.image = output_img
        output_img_ae_label.image = output_img_ae

        set_scales_to_latent_vec(scales, g_latent)

    def decode_new_latent(output_img, output_img_ae, scales):
        # need to build latent vector from scale values
        new_latent = [
            scale.get() for scale in scales
        ]  # scales are already 'normalized'

        output = decoder.predict([new_latent])[0]

        output_pil = Image.fromarray((output * 255).astype(np.uint8))
        output_blur = get_blur_factor(output_pil)

        output_ae = run_image_ae(ae, output_pil)
        output_ae_blur = get_blur_factor(output_ae)

        tk_img_output = ImageTk.PhotoImage(
            Image.fromarray((output * 255).astype(np.uint8))
        )
        tk_img_output_ae = ImageTk.PhotoImage(output_ae)

        text1 = f"Output, blur:{output_blur}"
        text2 = f"OutputAE, blur:{output_ae_blur}"
        output_img.configure(image=tk_img_output, text=text1)
        output_img.image = tk_img_output
        output_img.text = text1
        output_img_ae.configure(image=tk_img_output_ae, text=text2)
        output_img_ae.image = tk_img_output_ae
        output_img_ae.text = text2

    (
        img_default_input,
        img_default_output,
        img_default_output_ae,
        g_latent,
    ) = get_tk_imgs(encoder, decoder, ae, files[0])

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
    output_image_ae = Label(
        ws,
        justify=CENTER,
        compound=BOTTOM,
        pady=2,
        text="Output image AE",
        image=img_default_output_ae,
    )
    scales = [
        Scale(
            ws,
            from_=-5,
            to=5,
            resolution=1e-10,
            showvalue=0,
            command=lambda _: decode_new_latent(
                output_image_vae, output_image_ae, scales
            ),
        )
        for _ in range(g_latent.shape[0])
    ]

    set_scales_to_latent_vec(scales, g_latent)

    button = Button(
        ws,
        text="Change source image",
        command=lambda: change_img(
            input_image_vae, output_image_vae, output_image_ae, scales, len(files)
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
    output_image_ae.grid(row=3, column=0)
    # button2.grid(row=3, column=0)

    num_cols = 50
    for i, scale in enumerate(scales):
        scale.grid(row=(int(i / num_cols) % num_cols), column=((i + 1) % num_cols) + 1)

    ws.mainloop()


def main():
    encoder, decoder = get_vae("./models/bigVAE_64.h5")
    ae = get_ae("./models/convAE_accordion_v1.h5")
    files = glob.glob("./archive/images/*.jpg")[:100]
    gui(files, encoder, decoder, ae)


if __name__ == "__main__":
    main()
