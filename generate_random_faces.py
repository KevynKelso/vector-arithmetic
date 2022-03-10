# example of loading the generator model and generating images
from matplotlib import pyplot
from numpy import asarray, savez_compressed
from numpy.random import randint, randn
from tensorflow.keras.models import load_model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


# create a plot of generated images
def plot_generated(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis("off")
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :])
    pyplot.savefig("generated_faces.png")
    pyplot.close()


# load model
model = load_model("./rsync-receiver/generator_model_040.h5")
# generate points in latent space
latent_points = generate_latent_points(100, 100)
# save points
savez_compressed("latent_points.npz", latent_points)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# save plot
plot_generated(X, 10)
