import os
from pathlib import Path
import re
import glob
from typing import Tuple

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
    def __init__(self, save_path=None, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

        if save_path is None:
            self.save_dir = Path().resolve().parent / "examples/generated_images"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = save_path

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
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
        while len(checkpoint_files) > max_to_keep + 1:
            oldest_checkpoint = checkpoint_files.pop(0)
            full_path = os.path.join(checkpoint_dir, oldest_checkpoint)

            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")
            else:
                print(f"Skipped directory: {oldest_checkpoint}")


class GANCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, monitor='disc_loss', save_best_only=True, mode='min', verbose=1):
        super(GANCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if (self.mode == 'min' and current < self.best) or (
                self.mode == 'max' and current > self.best) or not self.save_best_only:
            if self.verbose > 0:
                print(
                    f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model')
            self.best = current

            # Save generator
            generator_path = os.path.join(self.checkpoint_dir,
                                          f'generator_epoch_{epoch + 1:02d}_{self.monitor}_{current:.2f}.keras')
            self.model.generator.save(generator_path)

            # Save discriminator
            discriminator_path = os.path.join(self.checkpoint_dir,
                                              f'discriminator_epoch_{epoch + 1:02d}_{self.monitor}_{current:.2f}.keras')
            self.model.discriminator.save(discriminator_path)

class Generator(Model):
    def __init__(self, input_dim: int, output_shape: Tuple[int, int, int]):
        super(Generator, self).__init__()
        self.model = self.build_generator(input_dim, output_shape)

    def build_generator(self, input_dim: int, output_shape: Tuple[int, int, int]):
        model = Sequential()

        # 7x7 image with 128 channels
        model.add(Dense(7 * 7 * 128, input_dim=input_dim))
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

        model.add(Conv2D(output_shape[-1], 4, padding='same', activation='tanh'))

        return model

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.model(inputs)


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(Discriminator, self).__init__()
        self.model = self.build_discriminator(input_shape)
    def build_discriminator(self, input_shape: Tuple[int, int, int]):
        model = Sequential()

        # First Conv Block
        model.add(Conv2D(32, 5, input_shape=input_shape))
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.model(inputs)

class VanillaGAN(Model):
    def __init__(self, generator=None, discriminator=None, latent_dim=None, output_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create attributes for gen and disc
        assert all(opt is not None for opt in (latent_dim, output_dim))
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.generator = Generator(latent_dim, output_dim) if generator is None else generator
        self.discriminator = Discriminator(output_dim) if discriminator is None else discriminator

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
        print("BATCH SIZE: ", len(batch))
        try:
            print("BATCH SSHAPE: ", batch.shape)
        except:
            pass

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

    def load_latest_checkpoint(self, checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'vanilla_gan_epoch_*.keras'))

        if not checkpoint_files:
            print("No checkpoints found in directory:", checkpoint_dir)
            return False

        # Sort checkpoints by epoch number
        checkpoint_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

        # Get the latest checkpoint file
        latest_checkpoint = checkpoint_files[-1]

        # Load the model from the checkpoint
        self.load_weights(latest_checkpoint)

        print(f"Loaded latest checkpoint: {latest_checkpoint}")
        return True

class ConditionalGAN(Model):
    def __init__(self, generator=None, discriminator=None, latent_dim=None, output_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create attributes for gen and disc
        assert all(opt is not None for opt in (latent_dim, output_dim))
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.generator = Generator(latent_dim, output_dim) if generator is None else generator
        self.discriminator = Discriminator(output_dim) if discriminator is None else discriminator

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

    def load_latest_checkpoint(self, checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'vanilla_gan_epoch_*.keras'))

        if not checkpoint_files:
            print("No checkpoints found in directory:", checkpoint_dir)
            return False

        # Sort checkpoints by epoch number
        checkpoint_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

        # Get the latest checkpoint file
        latest_checkpoint = checkpoint_files[-1]

        # Load the model from the checkpoint
        self.load_weights(latest_checkpoint)

        print(f"Loaded latest checkpoint: {latest_checkpoint}")
        return True
