import os
from pathlib import Path
import re
import glob
from typing import Tuple
import shutil
import io

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, Input, Embedding, \
    Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from matplotlib import pyplot as plt
import numpy as np


class ModelMonitor(Callback):
    def __init__(self, save_path=None, num_img=3, latent_dim=128, input_generators=None):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.input_generators = input_generators if input_generators is not None else [lambda:
                                                                                       tf.random.uniform((self.num_img,
                                                                                                          self.latent_dim,
                                                                                                          1))]
        if save_path is None:
            self.save_dir = Path().resolve().parent / "examples/generated_images"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = save_path

    def on_epoch_end(self, epoch, logs=None):
        inputs = [func() for func in self.input_generators]

        generated_images = self.model.generator(*inputs)
        generated_images.numpy()

        print("ModelMonitor: Saving image to ", self.save_dir)
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            label = f'l{int(inputs[1][i])}_' if len(inputs) > 1 else ""
            img.save(str(self.save_dir / f'generated_img_e{epoch}_{label}{i}.png'))


class CheckpointCleanupCallback(Callback):
    def __init__(self, checkpoint_dir, max_to_keep=1):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

    def on_epoch_end(self, epoch, logs=None):
        # After each epoch, remove old checkpoints
        self.remove_old_checkpoints(self.checkpoint_dir, self.max_to_keep)

    def remove_old_checkpoints(self, checkpoint_dir, max_to_keep=2):
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
    def __init__(self, checkpoint_dir, monitor='g_loss', save_best_only=True, mode='min', verbose=1):
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


class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)


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
        real_images = batch
        fake_images = self.generator(tf.random.normal((tf.shape(batch)[0], self.latent_dim)), training=False)

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
            gen_images = self.generator(tf.random.normal((tf.shape(batch)[0], self.latent_dim)), training=True)
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


class ConditionalGenerator(Model):
    def __init__(self, input_dim: int, nr_classes: int, output_shape: Tuple[int, int, int]):
        super(ConditionalGenerator, self).__init__()

        self.model = self._build_generator(input_dim, nr_classes, output_shape)

    def _build_generator(self, input_dim: int, nr_classes: int, output_shape: Tuple[int, int, int]):
        # label part / conditional part
        in_label = Input(shape=(1,))
        li = Embedding(nr_classes, 50)(in_label)  # embedding for categorical input
        n_nodes = 7 * 7
        li = Dense(n_nodes)(li)
        li = Reshape((7, 7, 1))(li)  # reshape to additional channel

        # latent vector/image gen input part
        in_lat = Input(shape=(input_dim,))
        n_nodes = 128 * 7 * 7
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((7, 7, 128))(gen)

        # merge image gen and label input
        merge = Concatenate()([gen, li])

        # upsample to 14x14
        gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              activation=LeakyReLU(alpha=0.2))(merge)
        gen = BatchNormalization()(gen)

        # upsample to 28x28
        gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              activation=LeakyReLU(alpha=0.2))(gen)
        gen = BatchNormalization()(gen)
        # output
        out_layer = Conv2D(output_shape[-1], (7, 7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

    def call(self, noise, labels):
        return self.model([noise, labels])


class ConditionalDiscriminator(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int, int, int], nr_classes: int, clip_value: float):
        super(ConditionalDiscriminator, self).__init__()
        self.model = self.build_discriminator(input_shape, nr_classes, clip_value)

    def build_discriminator(self, input_shape: Tuple[int, int, int], nr_classes: int, clip_value: float):
        # label part
        in_label = Input(shape=(1,))
        li = Embedding(nr_classes, 50)(in_label)
        n_nodes = input_shape[0] * input_shape[1]
        li = Dense(n_nodes)(li)
        li = Reshape((input_shape[0], input_shape[1], 1))(li)

        # img part
        in_image = Input(shape=input_shape)

        merge = Concatenate()([in_image, li])

        # downsample
        fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                    activation=LeakyReLU(alpha=0.2),
                    kernel_constraint=ClipConstraint(clip_value))(merge)
        fe = BatchNormalization()(fe)

        # downsample
        fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                    activation=LeakyReLU(alpha=0.2),
                    kernel_constraint=ClipConstraint(clip_value))(fe)
        fe = BatchNormalization()(fe)

        fe = Flatten()(fe)

        # use Wasserstein version
        out_layer = Dense(1, kernel_constraint=ClipConstraint(clip_value))(fe)

        model = Model([in_image, in_label], out_layer)

        return model

    def call(self, images, labels):
        return self.model([images, labels])


class ConditionalGAN(Model):
    def __init__(self, generator=None, discriminator=None, latent_dim=None, output_dim=None, nr_classes=None,
                 clip_value=0.01, nr_critic_training=5, log_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create attributes for gen and disc
        assert all(opt is not None for opt in (latent_dim, output_dim, nr_classes))
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.clip_value = clip_value
        self.nr_critic_training = nr_critic_training
        self.d_iter_counter = 0
        self.g_iter_counter = 0
        self.nr_classes = nr_classes
        if log_dir is None:
            log_dir = Path().resolve() / "logs/CGAN"
            if log_dir.exists():
                clear_dir = input(f"Log dir {log_dir} exists. Clear? Y/N")
                if clear_dir == "Y" or clear_dir == "y":
                    shutil.rmtree(log_dir)
                else:
                    print("You did not say Y to clearing the directory. Exiting now!")
                    exit()
            log_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = str(log_dir)
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

        self.generator = ConditionalGenerator(latent_dim, nr_classes, output_dim) if generator is None else generator
        self.discriminator = ConditionalDiscriminator(output_dim,
                                                      nr_classes,
                                                      clip_value) if discriminator is None else discriminator

    def compile(self, g_opt=None, d_opt=None, g_loss=None, d_loss=None, use_default=True, *args, **kwargs):
        super().compile(*args, **kwargs)

        if use_default:
            initial_learning_rate = 0.001
            g_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True
            )
            d_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True
            )
            self.g_opt = Adam(learning_rate=.00001)
            self.d_opt = Adam(learning_rate=.00005)
            self.g_loss = BinaryCrossentropy()
        else:
            assert all(opt is not None for opt in (g_opt, d_opt, g_loss, d_loss))
            self.g_opt = g_opt
            self.d_opt = d_opt
            self.g_loss = g_loss

    def train_step(self, batch):
        real_images, labels = batch
        batch_size = tf.shape(real_images)[0]

        # turn off training of generator. I think this is not necessary but for safety better to have
        self.generator.trainable = False
        self.discriminator.trainable = True

        for i in range(self.nr_critic_training):  # Train critic more times
            # Generate random noise
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Generate fake images
            generated_images = self.generator(random_latent_vectors, labels)

            with tf.GradientTape() as tape:
                real_output = self.discriminator(real_images, labels)
                fake_output = self.discriminator(generated_images, labels)

                # Wasserstein loss
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            tf.summary.scalar(f'critic_loss_{i}', d_loss, step=self.d_opt.iterations)

        # Freeze the discriminator
        self.generator.trainable = True
        self.discriminator.trainable = False
        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, labels)
            fake_output = self.discriminator(generated_images, labels)
            g_loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Log additional metrics
        tf.summary.scalar('generator_loss', g_loss, step=self.g_opt.iterations)
        tf.summary.scalar('wasserstein_distance', -d_loss, step=self.g_opt.iterations)

        self.log_generated_images(self.g_iter_counter)

        """if self.g_iter_counter % 5 == 0:
            with file_writer.as_default():
                tf.summary.image("Training data", img, step=self.g_opt.iterations)
            with self.file_writer.as_default():
                tf.summary.image('generated_images', generated_images , max_outputs=5,
                                 step=self.g_opt.iterations)
        """
        self.g_iter_counter += 1

        return {"d_loss": d_loss, "g_loss": g_loss}

    def log_generated_images(self, epoch):
        if True: #epoch % 5 == 0:
            num_examples = 5
            random_latent_vectors = tf.random.normal(shape=(num_examples, self.latent_dim))
            random_labels = tf.random.uniform(shape=(num_examples,), minval=0, maxval=self.nr_classes, dtype=tf.int32)
            generated_images = self.generator(random_latent_vectors, random_labels)

            # Convert labels to strings
            label_strings = [f"Class {label.numpy()}" for label in random_labels]

            # Create a figure with subplots
            fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
            for i, (img, label) in enumerate(zip(generated_images, label_strings)):
                axes[i].imshow(img[:, :, 0], cmap='gray')  # Assuming grayscale images
                axes[i].set_title(label)
                axes[i].axis('off')

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)

            # Log the image to TensorBoard
            with self.file_writer.as_default():
                tf.summary.image("Generated Images", image, step=epoch)

            plt.close(fig)

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
