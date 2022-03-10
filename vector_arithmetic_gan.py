import numpy as np
# example of loading the generator model and generating images
from matplotlib import pyplot
from numpy import asarray, expand_dims, load, mean, vstack
from numpy.random import randint, randn
from tensorflow.keras.models import load_model


# average list of latent space vectors
def average_points(points, ix):
    # convert to zero offset points
    zero_ix = [i - 1 for i in ix]
    # retrieve required points
    vectors = points[zero_ix]
    # average the vectors
    avg_vector = mean(vectors, axis=0)
    # combine original and avg vectors
    all_vectors = vstack((vectors, avg_vector))
    return all_vectors


# create a plot of generated images
def plot_generated(examples, rows, cols):
    # plot images
    for i in range(rows * cols):
        # define subplot
        pyplot.subplot(rows, cols, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :])
    pyplot.show()


# load model
model = load_model("./rsync-receiver/generator_model_040.h5")
# retrieve specific points
smiling_woman_ix = [61, 62, 27]
neutral_woman_ix = [16, 21, 49]
neutral_man_ix = [10, 31, 82]
# load the saved latent points
data = load("latent_points.npz")
points = data["arr_0"]
# average vectors
smiling_woman = average_points(points, smiling_woman_ix)
neutral_woman = average_points(points, neutral_woman_ix)
neutral_man = average_points(points, neutral_man_ix)
# combine all vectors
all_vectors = vstack((smiling_woman, neutral_woman, neutral_man))
# generate images
images = model.predict(all_vectors)
# scale pixel values
images = (images + 1) / 2.0
# plot_generated(images, 3, 4)

# smiling woman - neutral woman + neutral man = smiling man
smiling_feature = (
    np.min(smiling_woman[:-1], axis=0) + 3 - (np.min(neutral_woman[:-1], axis=0) + 3)
)
test1 = model.predict(expand_dims(np.min(smiling_woman[:-1] + 4, axis=0), 0))[0]
pyplot.imshow((test1 + 1) / 2.0)
pyplot.show()
test2 = model.predict(expand_dims(np.min(neutral_woman[:-1] + 4, axis=0), 0))[0]
pyplot.imshow((test2 + 1) / 2.0)
pyplot.show()

test3 = model.predict(expand_dims(smiling_feature, 0))[0]
pyplot.imshow((test3 + 1) / 2.0)
pyplot.show()


result_vector = neutral_man[-1] + smiling_feature
# generate image
result_vector = expand_dims(result_vector, 0)
result_image = model.predict(result_vector)
# scale pixel values
result_image = (result_image + 1) / 2.0
pyplot.imshow(result_image[0])
pyplot.show()
