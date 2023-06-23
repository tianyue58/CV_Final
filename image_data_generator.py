from augmentation import mix_up
from augmentation import cut_mix
from keras.preprocessing.image import ImageDataGenerator


class MixUpImageDataGenerator(ImageDataGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def mix_up_generator(self, generator):
        while True:
            batch_images, batch_labels = next(generator)
            mixed_images, mixed_labels = mix_up(batch_images, batch_labels, self.p)
            yield mixed_images, mixed_labels

    def flow(self, *args, **kwargs):
        generator = super().flow(*args, **kwargs)
        return self.mix_up_generator(generator)


class CutMixImageDataGenerator(ImageDataGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def cut_mix_generator(self, generator):
        while True:
            batch_images, batch_labels = next(generator)
            mixed_images, mixed_labels = cut_mix(batch_images, batch_labels, self.p)
            yield mixed_images, mixed_labels

    def flow(self, *args, **kwargs):
        generator = super().flow(*args, **kwargs)
        return self.cut_mix_generator(generator)


class CutOutImageDataGenerator(ImageDataGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p
