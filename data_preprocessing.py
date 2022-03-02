from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm

DIRECTORY = "rsync-receiver/"


def load_image(fname):
    image = Image.open(fname)
    image = image.convert("RGB")
    px = np.asarray(image)

    return px


# def load_faces(directory, n_faces):
# faces = list()
# for fname in listdir(directory):
# px = load_image(directory + fname)
# faces.append(px)
# if len(faces) >= n_faces:
# break

# return np.asarray(faces)


def plot_faces(faces, n):
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(faces[i])

    plt.show()


def extract_face(model, pixels, required_size=(80, 80)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return
    x1, y1, width, height = faces[0]["box"]
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels).resize(required_size)
    face_array = np.asarray(image)

    return face_array


def load_faces(directory, n_faces):
    model = MTCNN()
    faces = []

    for fname in tqdm(listdir(directory)):
        pixels = load_image(directory + fname)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
        if len(faces) >= n_faces:
            break

    return np.asarray(faces)


def load_cropped_faces():
    data = np.load("img_align_celeba.npz")
    faces = data["arr_0"]
    print("Loaded: ", faces.shape)

    return faces


def main():
    all_faces = load_faces(DIRECTORY, 26)
    print("Loaded: ", all_faces.shape)
    np.savez_compressed("img_align_celeba.npz", all_faces)
    faces = load_cropped_faces()
    plot_faces(faces, 5)


if __name__ == "__main__":
    main()
