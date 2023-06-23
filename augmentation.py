# https://github.com/beresandras/contrastive-classification-keras

import random
import numpy as np
import tensorflow as tf

from keras.layers import Layer


def mix_up(batch_images, batch_labels, p=0.5):
    batch_size = batch_images.shape[0]
    n_classes = batch_labels.shape[1]
    image_size = batch_images.shape[1]

    images = []
    labels = []
    for j in range(batch_size):
        prob = np.random.uniform(0, 1) <= p
        prob = float(prob)
        k = np.random.randint(0, batch_size)
        beta_dist = np.random.uniform(0, 1)
        weight = beta_dist * prob
        img1 = batch_images[j, :]
        img2 = batch_images[k, :]
        images.append((1 - weight) * img1 + weight * img2)

        lab1 = batch_labels[j, :]
        lab2 = batch_labels[k, :]
        labels.append((1 - weight) * lab1 + weight * lab2)

    output_images = np.reshape(np.stack(images), (batch_size, image_size, image_size, 3))
    output_labels = np.reshape(np.stack(labels), (batch_size, n_classes))
    return output_images, output_labels


def cut_mix(batch_images, batch_labels, p=0.5):
    batch_size = batch_images.shape[0]
    n_classes = batch_labels.shape[1]
    image_size = batch_images.shape[1]

    images = []
    labels = []
    for j in range(batch_size):
        prob = np.random.uniform(0, 1) <= p
        k = j
        while k == j:
            k = np.random.randint(0, batch_size)
        x = np.random.randint(0, image_size)
        y = np.random.randint(0, image_size)
        beta_dist = np.random.uniform(0, 1)
        width = int(image_size * np.sqrt(1 - beta_dist)) * prob
        ya = max(0, y - width // 2)
        yb = min(image_size, y + width // 2)
        xa = max(0, x - width // 2)
        xb = min(image_size, x + width // 2)

        one = batch_images[j, ya:yb, 0:xa, :]
        two = batch_images[k, ya:yb, xa:xb, :]
        three = batch_images[j, ya:yb, xb:image_size, :]
        middle = np.concatenate([one, two, three], axis=1)
        img = np.concatenate([batch_images[j, 0:ya, :, :], middle, batch_images[j, yb:image_size, :, :]], axis=0)
        images.append(img)

        weight = width * width / image_size / image_size
        lab1 = batch_labels[j]
        lab2 = batch_labels[k]
        labels.append((1 - weight) * lab1 + weight * lab2)

    output_images = np.reshape(np.stack(images), (batch_size, image_size, image_size, 3))
    output_labels = np.reshape(np.stack(labels), (batch_size, n_classes))
    return output_images, output_labels


def cut_out(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    img_h, img_w, img_c = image.shape
    p_1 = np.random.rand()
    if p_1 > p:
        return image

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    image[top:top + h, left:left + w, :] = c
    return image


# crops and resizes part of the image to the original resolutions
class RandomResizedCrop(Layer):
    def __init__(self, scale, ratio, **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
            random_ratios = tf.exp(tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1]))

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack([
                height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths,], axis=1)
            images = tf.image.crop_and_resize(images, bounding_boxes, tf.range(batch_size), (height, width))

        return images


# distorts the color distibutions of images
class RandomColorJitter(Layer):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)

        # color jitter ranges: (min jitter strength, max jitter strength)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # list of applicable color augmentations
        self.color_augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.random_hue
        ]
        # the tf.image.random_[brightness, contrast, saturation, hue] operations
        # cannot be used here, as they transform a batch of images in the same way

    def blend(self, images_1, images_2, ratios):
        # linear interpolation between two images, with values clipped to the valid range
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(images, 0, tf.random.uniform((tf.shape(images)[0], 1, 1, 1),
                                                       1 - self.brightness, 1 + self.brightness),)

    def random_contrast(self, images):
        # random interpolation/extrapolation between the image and its mean intensity value
        mean = tf.reduce_mean(tf.image.rgb_to_grayscale(images), axis=(1, 2), keepdims=True)
        return self.blend(images, mean, tf.random.uniform((tf.shape(images)[0], 1, 1, 1),
                                                          1 - self.contrast, 1 + self.contrast))

    def random_saturation(self, images):
        # random interpolation/extrapolation between the image and its grayscale counterpart
        return self.blend(images, tf.image.rgb_to_grayscale(images),
                          tf.random.uniform((tf.shape(images)[0], 1, 1, 1), 1 - self.saturation, 1 + self.saturation))

    def random_hue(self, images):
        # random shift in hue in hsv colorspace
        images = tf.image.rgb_to_hsv(images)
        images += tf.random.uniform((tf.shape(images)[0], 1, 1, 3), (-self.hue, 0, 0), (self.hue, 0, 0))
        # tf.math.floormod(images, 1.0) should be used here, however in introduces artifacts
        images = tf.where(images < 0.0, images + 1.0, images)
        images = tf.where(images > 1.0, images - 1.0, images)
        images = tf.image.hsv_to_rgb(images)
        return images

    def call(self, images, training=True):
        if training:
            # applies color augmentations in random order
            for color_augmentation in random.sample(self.color_augmentations, 4):
                images = color_augmentation(images)
        return images
