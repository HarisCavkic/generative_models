import os
from pathlib import Path

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
import numpy as np

class ModelMonitor(Callback):
    def __init__(self, save_path = None, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

        if save_path is None:
            self.save_dir = Path().resolve().parent / "examples/generated_images"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = save_path
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()


        print("ModelMonitor: Saving image to ", self.save_dir)
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(str(self.save_dir / f'generated_img_{epoch}_{i}.png'))

class CheckpointCleanupCallback(Callback):
    def __init__(self, checkpoint_dir, max_to_keep=1):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

    def on_epoch_end(self, epoch, logs=None):
        # After each epoch, remove old checkpoints
        self.remove_old_checkpoints(self.checkpoint_dir, self.max_to_keep)

    def remove_old_checkpoints(self, checkpoint_dir, max_to_keep=1):
        """Remove older checkpoints if there are more than max_to_keep."""
        # List all checkpoint files
        checkpoint_files = os.listdir(checkpoint_dir)

        # Sort the files to ensure chronological order
        checkpoint_files.sort()

        # If there are more checkpoints than max_to_keep, delete the oldest
        while len(checkpoint_files) > max_to_keep:
            # Remove the oldest file
            oldest_checkpoint = checkpoint_files.pop(0)
            os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
            print(f"Deleted old checkpoint: {oldest_checkpoint}")

class VanillaGAN(Model):
    def __init__(self, generator=None, discriminator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create attributes for gen and disc
        self.generator = generator if generator is not None else self.build_generator()
        self.discriminator = discriminator if discriminator is not None else self.build_discriminator()

    def compile(self, g_opt=None, d_opt=None, g_loss=None, d_loss=None, use_default=True, *args, **kwargs):
        super().compile(*args, **kwargs)

        if use_default:
            self.g_opt = Adam(learning_rate=0.0001)
            self.d_opt = Adam(learning_rate=0.00001)
            self.g_loss = BinaryCrossentropy()
            self.d_loss = BinaryCrossentropy()
        else:
            assert all(opt is not None for opt in (g_opt, d_opt, g_loss, d_loss))
            self.g_opt = g_opt
            self.d_opt = d_opt
            self.g_loss = g_loss
            self.d_loss = d_loss


    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # train discriminator
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat = tf.concat([yhat_real, yhat_fake], axis=0)

            y = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))

            y += tf.concat([noise_real, noise_fake], axis=0)

            total_d_loss = self.d_loss(y, yhat)

        # Apply backpropagation - nn learn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)
            predicted_labels = self.discriminator(gen_images, training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {'d_loss': total_d_loss, 'g_loss': total_g_loss}

    def build_generator(self):
        model = Sequential()

        model.add(Dense(7 * 7 * 128, input_dim=128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

        return model

    def build_discriminator(self):
        model = Sequential()

        # First Conv Block
        model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Second Conv Block
        model.add(Conv2D(64, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Third Conv Block
        model.add(Conv2D(128, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Fourth Conv Block
        model.add(Conv2D(256, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Flatten then pass to dense layer
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        return model
