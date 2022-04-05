from glob import glob

import imageio
from tqdm import tqdm


def main():
    files = glob("./images/*.png")
    images = []
    for f in tqdm(files):
        images.append(imageio.imread(f))
    imageio.mimsave("./pretrained_ae_gan.gif", images, duration=0.5)


if __name__ == "__main__":
    main()
