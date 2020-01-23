import numpy as np
import skimage

from skimage.transform import radon, iradon
from skimage.draw import ellipse, rectangle
from skimage.transform import rotate


def generate_training_data(image_size, number_of_images, number_of_angles):

    training_images = []

    min_radius = image_size / 32
    max_radius = image_size / 8
    shape = (image_size, image_size)

    for i in range(number_of_images):
        image = np.zeros(shape=shape)

        n_ellipses = np.random.randint(10, 20)
        center = np.random.normal(image_size / 2, image_size / 4, size=(n_ellipses, 2))
        radius_r = np.random.randint(min_radius, max_radius + 1, size=n_ellipses)
        radius_c = np.random.randint(min_radius / 2, max_radius / 2 + 1, size=n_ellipses)
        rotation = np.random.uniform(-np.pi, np.pi, size=n_ellipses)

        for k in range(n_ellipses):
            xp, yp = ellipse(r=center[k, 0], c=center[k, 1], r_radius=radius_r[k], c_radius=radius_c[k], shape=shape, rotation=rotation[k])
            image[xp, yp] += np.random.uniform(0., 10.)

        training_images.append(image)

    training_images = np.asarray(training_images) / 255.

    inverted_images = []
    th = np.linspace(0., 180., number_of_angles, endpoint=False)

    for k in range(number_of_images):
        inverted_images.append(iradon(radon(training_images[k], theta=th, circle=False), circle=False))

    inverted_images = np.asarray(inverted_images)

    return np.expand_dims(training_images, axis=-1), np.expand_dims(inverted_images, axis=-1)


def generate_testing_data(image_size, number_of_images, number_of_angles):

    validation_images = []

    min_size = image_size // 16
    max_size = image_size // 4
    shape = (image_size, image_size)

    for i in range(number_of_images):
        image = np.zeros(shape=shape)

        n_rectangles = np.random.randint(10, 20)
        start = np.random.normal(image_size / 2, image_size / 4, size=(n_rectangles, 2)).astype(int)
        height = np.random.randint(min_size, max_size + 1, size=n_rectangles)
        width = np.random.randint(min_size, max_size + 1, size=n_rectangles)
        rotation = np.random.uniform(-np.pi, np.pi, size=n_rectangles)

        for k in range(n_rectangles):
            xp, yp = rectangle(start=(start[k,0], start[k,1]), end=(start[k,0]+width[k], start[k,1]+height[k]), shape=shape)
            image_pre_rotation = np.zeros(shape=shape)
            image_pre_rotation[xp, yp] = np.random.uniform(0., 10.)
            image += rotate(image_pre_rotation, angle=rotation[k])

        validation_images.append(image)

    validation_images = np.asarray(validation_images) / 255.

    inverted_images = []
    th = np.linspace(0., 180., number_of_angles, endpoint=False)

    for k in range(number_of_images):
        inverted_images.append(iradon(radon(validation_images[k], theta=th, circle=False), circle=False))

    inverted_images = np.asarray(inverted_images)

    return np.expand_dims(validation_images, axis=-1), np.expand_dims(inverted_images, axis=-1)


def print_data_samples(Y, X, n_samples):
    """
    Prints n_samples pair of images (Y, X), where:
    Y --- ground truth
    X --- noisy approximation of Y
    """
    plt.figure(figsize=(3. * n_samples, 6.))

    for i in range(1, n_samples + 1):
        plt.subplot(n_samples, 2, i)
        plt.imshow(Y[i - 1])
        plt.axis('off')
        plt.subplot(n_samples, 2, i + n_samples)
        plt.imshow(X[i - 1])
        plt.axis('off')

    plt.show()


def print_model_outputs(model, Y, X, n_samples):
    """
    Prints n_samples triple of images (Y, X, P), where:
    Y --- ground truth
    X --- noisy approximation of Y
    P = model(X) --- cleaned X
    """
    P = model(X, training=False)

    plt.figure(figsize=(3. * n_samples, 9.))

    for i in range(1, n_samples + 1):
        plt.subplot(n_samples, 3, i)
        plt.imshow(Y[i - 1])
        plt.axis('off')
        plt.subplot(n_samples, 3, i + n_samples)
        plt.imshow(X[i - 1])
        plt.axis('off')
        plt.subplot(n_samples, 3, i + 2 * n_samples)
        plt.imshow(P[i - 1])
        plt.axis('off')

    plt.show()
