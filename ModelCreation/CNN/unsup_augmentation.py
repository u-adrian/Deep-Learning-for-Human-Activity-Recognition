import tsaug
import numpy as np
import pandas as pd


def prep_for_tsaug(x_train, y_train):

    # converting into DataFrame for preprocessing
    df_x_train = pd.DataFrame(x_train)
    df_y_train = pd.DataFrame(y_train)

    # concatenating x and y again as a basis to restructure the dataset
    df_x_y_train = pd.concat([df_x_train, df_y_train], axis=1)

    # splitting dataframes into smaller ones which possess the same label
    # splitting the dataframes again into x and y format
    # converting dataframes into numpy arrays
    list_of_labels = np.sort(df_x_y_train.iloc[:, -1].unique())

    x_train_tsaug_list = []
    y_train_tsaug_list = []

    for label in list_of_labels:
        df = df_x_y_train[df_x_y_train.iloc[:, -1] == label]

        df_x_train = df.iloc[:, :-1]
        df_y_train = df.iloc[:, -1]

        x_train_tsaug_list.append(np.array(df_x_train))
        y_train_tsaug_list.append(np.array(df_y_train))

        # the amount of datapoints per label varies, therefore we cannot input all of the data in go into 	tsaug augmentations. Instead we apply the augmentation on the individual arrays per label.
        # print(f"Shape of time series for label {label}:", "\tx_train ->", np.array(df_x_train).shape, "\t 	y_train ->", np.array(df_y_train).shape)

    return x_train_tsaug_list, y_train_tsaug_list


def aug(x_train, y_train):
    x_train_tsaug, y_train_tsaug = prep_for_tsaug(x_train, y_train)
    augmenter = tsaug.AddNoise(scale=0.001)
    test = []
    for i, array in enumerate(x_train_tsaug):
        # aug_array_x = augmenter.augment(array)
        test.append(array)
    x_train_aug_comp = np.vstack(test)
    return x_train_aug_comp, np.vstack(y_train_tsaug)


def batch_aug(x_train_batch, augmenter=None):
    x_train_batch = x_train_batch[:, 0, :, :]
    if augmenter is None:
        transforms = [
            tsaug.AddNoise(scale=0.003) @ 0.1,
            tsaug.Crop(size=25) @ 0.1,
            tsaug.Quantize(n_levels=30) @ 0.1,
            tsaug.Drift(max_drift=(0.01, 0.03)) @ 0.1,
            tsaug.Reverse() @ 0.1,
        ]
        augmenter = transforms[0]
    x_train_aug = augmenter.augment(x_train_batch)
    return x_train_aug[:, np.newaxis, :, :]


def rand_aug(x_train_batch, N=4, M=3):
    transforms = [
        tsaug.AddNoise(scale=0.001 * M) @ 0.1,
        tsaug.Crop(size=25) @ 0.1,
        tsaug.Quantize(n_levels=10 * M) @ 0.1,
        tsaug.Drift(max_drift=(0.01, 0.03)) @ 0.1,
        tsaug.Reverse() @ 0.1,
    ]
    x_train_aug = x_train_batch[:, 0, :, :]
    sampled_ops = np.random.choice(transforms, N)
    for op in sampled_ops:
        x_train_aug = op.augment(x_train_aug)
    return x_train_aug[:, np.newaxis, :, :]
