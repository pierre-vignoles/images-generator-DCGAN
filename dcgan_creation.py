from typing import Tuple
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, LeakyReLU, Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset

BUFFER_SIZE: int = 5000
BATCH_SIZE: int = 64
noise_dim: int = 100
img_rows: int = 32
img_cols: int = 32
channels: int = 3
EPOCHS: int = 200


# load and prepare cifar10 training images
def load_real_samples(BUFFER_SIZE: int, BATCH_SIZE: int) -> Tuple[np.ndarray, PrefetchDataset]:
    (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train[np.where(y_train == 3)[0]]
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels).astype('float32')
    X_train = (X_train - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # creation of a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=BUFFER_SIZE)  # (shuffle => repeat => batch)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1) # prefech (add in cache x batch during processing an other one)
    return X_train, dataset


def define_generator(noise_dim: int) -> Sequential:
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Reshape((4, 4, 256)))
    # 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model


def define_discriminator(in_shape=(32,32,3)) -> Sequential:
    model = Sequential()
    # 32x32
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # 16x16
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # 8x8
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # 4x4
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_gan(g_model: Sequential, d_model: Sequential) -> Sequential:
    # make weights in the discriminator not trainable
    d_model.trainable = False
    model = Sequential([g_model, d_model])
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# Generate images based just on noise
def generation_images(BATCH_SIZE: int, noise_dim: int, generator: Sequential) -> tf.Tensor:
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    generated_image = generator(noise, training=True)
    return generated_image


# Save and plot images
def generate_and_save_images(static_noise: tf.Tensor, epoch: int, generator: Sequential):
    generated_images = generator(static_noise, training=False)
    plt.figure(figsize=(10, 10))

    for i in range(6 * 6):
        plt.subplot(6, 6, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('img/image_at_epoch_{:04d}.png'.format(epoch + 1))
    plt.show()


def train_model(EPOCHS: int, BATCH_SIZE: int, X_train: np.ndarray, dataset: PrefetchDataset, noise_dim: int,
                static_noise: tf.Tensor, model: Sequential):
    generator, discriminator = model.layers

    for epoch in range(EPOCHS):
        print(f"Currently on Epoch {epoch + 1}")
        i = 0
        # For every batch in the dataset
        for X_batch in dataset:
            i = i + 1
            # show where the algo is
            if i % 100 == 0:
                print(f"\tCurrently on batch number {i} of {len(X_train) // BATCH_SIZE}")

            ###### TRAINING THE DISCRIMINATOR

            # Generate images based just on noise input
            gen_images = generation_images(BATCH_SIZE, noise_dim, generator)

            # Concatenate Generated Images against the Real Ones
            X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis=0)

            # Set to zero for fake images and 0.9 for real images
            y1 = tf.constant([[0.]] * BATCH_SIZE + [[0.9]] * BATCH_SIZE)

            # Train the discriminator on this batch
            d_loss = discriminator.train_on_batch(X_fake_vs_real, y1)

            ###### TRAINING THE GENERATOR

            # Create some noise
            noise = tf.random.normal(shape=[BATCH_SIZE, noise_dim])

            # We want discriminator to believe that fake images are real
            y2 = tf.constant([[1.]] * BATCH_SIZE)

            g_loss = model.train_on_batch(noise, y2)

        # save image every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'\t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
            generate_and_save_images(static_noise, epoch, generator)
        # Save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            filename = 'models/generator_model_%03d.h5' % (epoch + 1)
            generator.save(filename)


def main_dcgan():
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(noise_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # load data
    X_train, dataset = load_real_samples(BUFFER_SIZE, BATCH_SIZE)
    # static noise
    static_noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # train model
    train_model(EPOCHS, BATCH_SIZE, X_train, dataset, noise_dim, static_noise, gan_model)