import tensorflow_datasets as tfds

from custom_GANs.simple_GANs import VanillaGAN, ModelMonitor


def scale_images(data):
    image = data['image']
    return image / 255

def main():
    gan = VanillaGAN()

    ds = tfds.load('fashion_mnist', split='train')
    ds = ds.map(scale_images)
    ds = ds.cache()
    ds = ds.shuffle(60000)
    ds = ds.batch(128)
    ds = ds.prefetch(64)

    gan.compile(use_default=True)
    hist = gan.fit(ds, epochs = 5, callbacks = [ModelMonitor()])



if __name__ == '__main__':
    main()