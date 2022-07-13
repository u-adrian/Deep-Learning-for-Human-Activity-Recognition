import math

import numpy as np

import augmentation

import tsaug


def do_augmentations(batch_x, batch_y):
    b_x, b_y = aug_scaling(batch_x, batch_y, 0.3)
    b_x, b_y = aug_jitter(b_x, b_y, 0.3)
    # print("Before:", b_x.shape)
    b_x, _ = aug_time_warp(b_x, b_y, 0.5)
    b_x, _ = aug_noise(b_x, b_y, 0.5)
    b_x, _ = aug_convolve(b_x, b_y, 0.5)
    b_x, _ = aug_crop(b_x, b_y, 0.5)
    b_x, _ = aug_drift(b_x, b_y, 0.5)
    b_x, _ = aug_dropout(b_x, b_y, 0.5)
    b_x, _ = aug_pool(b_x, b_y, 0.5)
    b_x, _ = aug_quantize(b_x, b_y, 0.5)
    # print("After:", b_x.shape)
    return b_x, b_y


def aug_jitter(batch_x, batch_y, prob):
    function = lambda x: augmentation.jitter(x=x, sigma=0.02)
    return augmentation_capsule(batch_x, batch_y, function=function, prob=prob)


def aug_scaling(batch_x, batch_y, prob):
    function = lambda x: augmentation.scaling(x=x)
    return augmentation_capsule(batch_x, batch_y, function=function, prob=prob)


def aug_time_warp(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=2, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_noise(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.AddNoise(scale=0.01, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_convolve(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Convolve(window="flattop", size=11, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_crop(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Crop(size=batch_x.shape[2], prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_drift(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Drift(max_drift=0.7, n_drift_points=5, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_dropout(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Dropout(
        p=0.1, size=(1, 3), fill=float(0.0), per_channel=True, prob=prob
    ).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_pool(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Pool(kind="max", size=3, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def aug_quantize(batch_x, batch_y, prob):
    b_x = np.squeeze(batch_x)
    b_x = tsaug.Quantize(n_levels=20, prob=prob).augment(b_x)
    b_x = np.expand_dims(b_x, axis=1)
    return b_x, batch_y


def augmentation_capsule(batch_x, batch_y, function, prob):
    b_x = np.transpose(batch_x, axes=[0, 1, 3, 2])
    b_x = np.squeeze(b_x)

    batch_size = batch_x.shape[0]
    rand_vec = np.expand_dims((np.random.random(batch_size) <= prob), axis=[1, 2])

    augmented_batch = function(x=b_x)

    b_x = rand_vec * augmented_batch + (1 - rand_vec) * b_x

    b_x = np.expand_dims(b_x, axis=1)
    b_x = np.transpose(b_x, axes=[0, 1, 3, 2])
    return b_x, batch_y


def time_slot_disturbance(batch_x):
    batch_size = batch_x.shape[0]
    window_size = batch_x.shape[2]
    slot_size = 3
    b_x = batch_x
    start_point = np.floor(
        np.random.random(batch_size) * (window_size - slot_size)
    ).astype(np.int)
    # print(start_point)
    b_x[:, :, start_point] = 0
    print(b_x[:, :, start_point].shape)
    return b_x
