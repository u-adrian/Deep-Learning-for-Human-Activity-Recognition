import math

import numpy as np

import augmentation


def do_augmentations(batch_x, batch_y):
    b_x, b_y = aug_scaling(batch_x, batch_y, 0.3)
    b_x, b_y = aug_jitter(b_x, b_y, 0.3)
    #b_x, b_y = aug_time_warp(b_x, b_y, 0.3)
    b_x = time_slot_disturbance(b_x)
    return b_x, b_y


def aug_jitter(batch_x, batch_y, prob):
    function = lambda x : augmentation.jitter(x=x, sigma=0.02)
    return augmentation_capsule(batch_x, batch_y, function=function, prob=prob)


def aug_scaling(batch_x, batch_y, prob):
    function = lambda x: augmentation.scaling(x=x)
    return augmentation_capsule(batch_x, batch_y, function=function, prob=prob)


def aug_time_warp(batch_x, batch_y, prob):
    function = lambda x: augmentation.time_warp(x=x)
    return augmentation_capsule(batch_x, batch_y, function=function, prob=prob)


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
    start_point = np.floor(np.random.random(batch_size) * (window_size - slot_size)).astype(np.int)
    #print(start_point)
    b_x[:,:,start_point] = 0
    print(b_x[:,:,start_point].shape)
    return b_x


