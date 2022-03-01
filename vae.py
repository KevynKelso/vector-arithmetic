import numpy as np


def load_cropped_faces():
    data = np.load("img_align_celeba.npz")
    faces = data["arr_0"]
    print("Loaded: ", faces.shape)

    return faces


def main():
    print(load_cropped_faces())


if __name__ == "__main__":
    main()
