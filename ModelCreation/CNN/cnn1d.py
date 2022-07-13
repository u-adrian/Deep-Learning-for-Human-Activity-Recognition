import argparse
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf
import time
from sklearn import metrics
from sklearn.model_selection import KFold
import h5py
import os
import sys
import matplotlib.pyplot as plt
import math

from unsup_augmentations import prep_for_tsaug, aug, batch_aug, rand_aug

tf.disable_v2_behavior()

#%matplotlib inline
plt.style.use("ggplot")


# FUNCTION DECLARATION


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries_" + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean_" + name, mean)
        tf.summary.scalar(name + "_value", var)
        tf.summary.histogram("histogram_" + name, var)


def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += math.floor(size / 2)  # TODO


def segment_opp(x_train, y_train, window_size):
    segments = np.zeros(((len(x_train) // (window_size // 2)) - 1, window_size, 77))
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if len(x_train[start:end]) == window_size:
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def segment_dap(x_train, y_train, window_size):
    segments = np.zeros(((len(x_train) // (window_size // 2)) - 1, window_size, 9))
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if len(x_train[start:end]) == window_size:
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def segment_pa2(x_train, y_train, window_size):
    segments = np.zeros(
        ((len(x_train) // (window_size // 2)) - 1, window_size, 52)
    )  # shape ist: (#windows, window len, #labels)
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if len(x_train[start:end]) == window_size:
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def segment_sph(x_train, y_train, window_size):
    segments = np.zeros(((len(x_train) // (window_size // 2)) - 1, window_size, 52))
    labels = np.zeros(((len(y_train) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windowz(x_train, window_size):
        if len(x_train[start:end]) == window_size:
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def depth_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(
        x, ksize=[1, 1, kernel_size, 1], strides=[1, 1, stride_size, 1], padding="VALID"
    )


def get_data(dataset, data_path, num_channels):
    if dataset == "opp":
        path = os.path.join(data_path, "OpportunityUCIDataset", "opportunity.h5")
    elif dataset == "dap":
        path = os.path.join(data_path, "dataset_fog_release", "daphnet.h5")
    elif dataset == "pa2":
        path = os.path.join(data_path, "PAMAP2_Dataset", "pamap2.h5")
    elif dataset == "sph":
        path = os.path.join(data_path, "SphereDataset", "sphere.h5")
    else:
        print("Dataset not supported yet")
        sys.exit()

    f = h5py.File(path, "r")

    x_train = f.get("train").get("inputs")[()]
    y_train = f.get("train").get("targets")[()]

    x_test = f.get("test").get("inputs")[()]
    y_test = f.get("test").get("targets")[()]

    print("x_train shape = ", x_train.shape)
    print("y_train shape =", y_train.shape)
    print("x_test shape =", x_test.shape)
    print("y_test shape =", y_test.shape)

    if dataset == "dap":
        # downsample to 30 Hz
        x_train = x_train[::2, :]
        y_train = y_train[::2]
        x_test = x_test[::2, :]
        y_test = y_test[::2]
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)

    if dataset == "pa2":
        # downsample to 30 Hz
        x_train = x_train[::3, :]
        y_train = y_train[::3]
        x_test = x_test[::3, :]
        y_test = y_test[::3]
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)

    print(np.unique(y_train))
    print(np.unique(y_test))

    input_width = 23
    if dataset == "opp":
        input_width = 23
        print("segmenting signal...")
        train_x, train_y = segment_opp(x_train, y_train, input_width)
        test_x, test_y = segment_opp(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "dap":
        print("dap seg")
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_dap(x_train, y_train, input_width)
        test_x, test_y = segment_dap(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "pa2":
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_pa2(x_train, y_train, input_width)
        test_x, test_y = segment_pa2(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "sph":
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_sph(x_train, y_train, input_width)
        test_x, test_y = segment_sph(x_test, y_test, input_width)
        print("signal segmented.")
    else:
        print("no correct dataset")

    print("train_x shape =", train_x.shape)
    print("train_y shape =", train_y.shape)
    print("test_x shape =", test_x.shape)
    print("test_y shape =", test_y.shape)

    # http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/

    train = pd.get_dummies(train_y)
    test = pd.get_dummies(test_y)

    train, test = train.align(test, join="inner", axis=1)  # maybe 'outer' is better

    train_y = np.asarray(train)
    test_y = np.asarray(test)

    print("unique test_y", np.unique(test_y))
    print("unique train_y", np.unique(train_y))
    print("test_y[1]=", test_y[1])
    # test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
    print("train_y shape(1-hot) =", train_y.shape)
    print("test_y shape(1-hot) =", test_y.shape)

    train_x = train_x.reshape(len(train_x), 1, input_width, num_channels)  # opportunity
    test_x = test_x.reshape(len(test_x), 1, input_width, num_channels)  # opportunity
    print("train_x_reshaped = ", train_x.shape)
    print("test_x_reshaped = ", test_x.shape)
    print("train_x shape =", train_x.shape)
    print("train_y shape =", train_y.shape)
    print("test_x shape =", test_x.shape)
    print("test_y shape =", test_y.shape)

    return train_x, train_y, test_x, test_y


def get_model(X, num_channels, num_labels):
    stride_size = 2
    kernel_size_1 = 7
    kernel_size_2 = 3
    kernel_size_3 = 1
    depth_1 = 128
    depth_2 = 128
    depth_3 = 128
    num_hidden = 512  # neurons in the fully connected layer

    dropout_1 = tf.placeholder(tf.float32)  # 0.1
    dropout_2 = tf.placeholder(tf.float32)  # 0.25
    dropout_3 = tf.placeholder(tf.float32)  # 0.5

    # HIDDEN LAYERS AND FULLY CONNECTED FOR Opportunity etc
    # https://www.tensorflow.org/get_started/mnist/pros

    # hidden layer 1
    W_conv1 = weight_variable([1, kernel_size_1, num_channels, depth_1])
    b_conv1 = bias_variable([depth_1])

    h_conv1 = tf.nn.relu(depth_conv2d(X, W_conv1) + b_conv1)
    h_conv1 = tf.nn.dropout(h_conv1, dropout_1)

    h_pool1 = max_pool(h_conv1, kernel_size_1, stride_size)

    # hidden layer 2
    W_conv2 = weight_variable([1, kernel_size_2, depth_1, depth_2])
    b_conv2 = bias_variable([depth_2])

    h_conv2 = tf.nn.relu(depth_conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, dropout_2)

    h_pool2 = max_pool(h_conv2, kernel_size_2, stride_size)

    # first we get the shape of the last layer and flatten it out
    shape = h_pool2.get_shape().as_list()

    W_fc1 = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
    b_fc1 = bias_variable([num_hidden])

    h_pool3_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, dropout_3)

    # readout layer.

    W_fc2 = weight_variable([num_hidden, num_labels])
    b_fc2 = bias_variable([num_labels])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv, dropout_1, dropout_2, dropout_3


def cnn_execute(dataset, data_path, aug_function=None):
    # MAIN ()

    print("starting...")
    start_time = time.time()

    # DATA PREPROCESSING
    if dataset == "opp":
        print("opp")
        input_height = 1
        input_width = 23  # or 90 for actitracker
        num_labels = 18  # or 6 for actitracker
        num_channels = 77  # or 3 for actitracker
    elif dataset == "dap":
        print("dap")
        input_height = 1
        input_width = 25  # or 90 for actitracker
        num_labels = 2  # or 6 for actitracker
        num_channels = 9  # or 3 for actitracker
    elif dataset == "pa2":
        print("pa2")
        input_height = 1
        input_width = 25  # or 90 for actitracker
        num_labels = 11  # or 6 for actitracker
        num_channels = 52  # or 3 for actitracker
    elif dataset == "sph":
        print("sph")
        input_height = 1
        input_width = 25  # or 90 for actitracker
        num_labels = 20  # or 6 for actitracker
        num_channels = 52  # or 3 for actitracker
    else:
        print("wrong dataset")
        sys.exit()

    train_x, train_y, test_x, test_y = get_data(dataset, data_path, num_channels)

    data_x, data_y = np.concatenate([train_x, test_x]), np.concatenate(
        [train_y, test_y]
    )

    X = tf.placeholder(
        tf.float32, shape=[None, input_height, input_width, num_channels]
    )
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    print("X shape =", X.shape)
    print("Y shape =", Y.shape)

    y_conv, dropout_1, dropout_2, dropout_3 = get_model(X, num_channels, num_labels)

    batch_size = 64

    learning_rate = 0.0005
    training_epochs = 50

    # TRAINING THE MODEL
    config = tf.ConfigProto(device_count={"GPU": 0})

    loss_dicts = []
    score_dicts = []
    confusion_matrices = []

    n_splits = 5

    kfold = KFold(n_splits=n_splits, shuffle=True)
    with tf.Session(config=config) as session:

        for k, (train_index, test_index) in enumerate(kfold.split(data_x)):
            print(f"Kfold: start Split {k+1} of {n_splits} Splits")

            train_x, test_x = data_x[train_index], data_x[test_index]
            train_y, test_y = data_y[train_index], data_y[test_index]

            # COST FUNCTION
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv)
            )

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss
            )

            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            loss_dict = {
                "Train_Accuracy": [],
                "Train_Loss": [],
                "Test_Accuracy": [],
            }

            tf.compat.v1.initialize_all_variables().run()

            total_batches = train_x.shape[0] // batch_size
            for epoch in range(training_epochs):

                cost_history = np.empty(shape=[0], dtype=float)

                for b in range(total_batches):
                    offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                    batch_x = train_x[offset : (offset + batch_size), :, :, :]
                    batch_y = train_y[offset : (offset + batch_size), :]

                    if aug_function is not None:
                        batch_x, batch_y = aug_function(batch_x, batch_y)

                    _, c = session.run(
                        [optimizer, loss],
                        feed_dict={
                            X: batch_x,
                            Y: batch_y,
                            dropout_1: 1 - 0.1,
                            dropout_2: 1 - 0.25,
                            dropout_3: 1 - 0.5,
                        },
                    )
                    cost_history = np.append(cost_history, c)
                mean_train_loss = np.mean(cost_history)
                train_accuracy = session.run(
                    accuracy,
                    feed_dict={
                        X: train_x,
                        Y: train_y,
                        dropout_1: 1 - 0.1,
                        dropout_2: 1 - 0.25,
                        dropout_3: 1 - 0.5,
                    },
                )
                test_accuracy = session.run(
                    accuracy,
                    feed_dict={
                        X: test_x,
                        Y: test_y,
                        dropout_1: 1,
                        dropout_2: 1,
                        dropout_3: 1,
                    },
                )

                loss_dict["Train_Loss"].append(str(mean_train_loss))
                loss_dict["Train_Accuracy"].append(str(train_accuracy))
                loss_dict["Test_Accuracy"].append(str(test_accuracy))

                print(
                    "Epoch: ",
                    epoch,
                    " Training Loss: ",
                    mean_train_loss,
                    " Training Accuracy: ",
                    train_accuracy,
                    "Testing Accuracy:",
                    test_accuracy,
                )

            y_p = tf.argmax(y_conv, 1)
            val_accuracy, y_pred = session.run(
                [accuracy, y_p],
                feed_dict={
                    X: test_x,
                    Y: test_y,
                    dropout_1: 1,
                    dropout_2: 1,
                    dropout_3: 1,
                },
            )
            print("validation accuracy:", val_accuracy)
            y_true = np.argmax(test_y, 1)

            score_dict = {"f1_score_w": [], "f1_score_m": [], "f1_score_mean": []}
            if dataset == "opp" or dataset == "pa2":
                score_dict["f1_score_w"].append(
                    metrics.f1_score(y_true, y_pred, average="weighted")
                )
                score_dict["f1_score_m"].append(
                    metrics.f1_score(y_true, y_pred, average="macro")
                )
                # print "f1_score_mean", metrics.f1_score(y_true, y_pred, average="micro")
                print("f1_score_w", score_dict["f1_score_w"][-1])

                print("f1_score_m", score_dict["f1_score_m"][-1])
                # print "f1_score_per_class", metrics.f1_score(y_true, y_pred, average=None)
            elif dataset == "dap":
                score_dict["f1_score_m"].append(
                    metrics.f1_score(y_true, y_pred, average="macro")
                )
                print("f1_score_m", score_dict["f1_score_m"][-1])
            elif dataset == "sph":
                score_dict["f1_score_mean"].append(
                    metrics.f1_score(y_true, y_pred, average="micro")
                )
                score_dict["f1_score_w"].append(
                    metrics.f1_score(y_true, y_pred, average="weighted")
                )
                score_dict["f1_score_w"].append(
                    metrics.f1_score(y_true, y_pred, average="weighted")
                )
                score_dict["f1_score_m"].append(
                    metrics.f1_score(y_true, y_pred, average="macro")
                )

                print("f1_score_mean", score_dict["f1_score_mean"][-1])
                print("f1_score_w", score_dict["f1_score_w"][-1])
                print("f1_score_m", score_dict["f1_score_m"][-1])
            else:
                print("wrong dataset")

            print("confusion_matrix")
            confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
            print(confusion_matrix)

            #######################################################################################
            #### micro- macro- weighted explanation ###############################################
            #                                                                                     #
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html      #
            #                                                                                     #
            # micro :Calculate metrics globally by counting the total true positives,             #
            # false negatives and false positives.                                                #
            #                                                                                     #
            # macro :Calculate metrics for each label, and find their unweighted mean.            #
            # This does not take label imbalance into account.                                    #
            #                                                                                     #
            # weighted :Calculate metrics for each label, and find their average, weighted        #
            # by support (the number of true instances for each label). This alters macro         #
            # to account for label imbalance; it can result in an F-score that is not between     #
            # precision and recall.                                                               #
            #                                                                                     #
            #######################################################################################

            print("--- %s seconds ---" % (time.time() - start_time))
            print("done.")

            loss_dicts.append(loss_dict)
            score_dicts.append(score_dict)
            confusion_matrices.append(confusion_matrix)

    return loss_dicts, score_dicts, confusion_matrices


def label_counting(dataset_name, data_path):
    # MAIN ()

    print("starting...")
    start_time = time.time()

    # DATA PREPROCESSING

    dataset = dataset_name
    if dataset == "opp":
        path = os.path.join(data_path, "OpportunityUCIDataset", "opportunity.h5")
    elif dataset == "dap":
        path = os.path.join(data_path, "dataset_fog_release", "daphnet.h5")
    elif dataset == "pa2":
        path = os.path.join(data_path, "PAMAP2_Dataset", "pamap2.h5")
    elif dataset == "sph":
        path = os.path.join(data_path, "SphereDataset", "sphere.h5")
    else:
        print("Dataset not supported yet")
        sys.exit()

    f = h5py.File(path, "r")

    x_train = f.get("train").get("inputs")[()]
    y_train = f.get("train").get("targets")[()]

    x_test = f.get("test").get("inputs")[()]
    y_test = f.get("test").get("targets")[()]

    print("x_train shape = ", x_train.shape)
    print("y_train shape =", y_train.shape)
    print("x_test shape =", x_test.shape)
    print("y_test shape =", y_test.shape)

    if dataset == "dap":
        # downsample to 30 Hz
        x_train = x_train[::2, :]
        y_train = y_train[::2]
        x_test = x_test[::2, :]
        y_test = y_test[::2]
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)

    if dataset == "pa2":
        # downsample to 30 Hz
        x_train = x_train[::3, :]
        y_train = y_train[::3]
        x_test = x_test[::3, :]
        y_test = y_test[::3]
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)

    print(np.unique(y_train))
    print(np.unique(y_test))

    input_width = 23
    if dataset == "opp":
        input_width = 23
        print("segmenting signal...")
        train_x, train_y = segment_opp(x_train, y_train, input_width)
        test_x, test_y = segment_opp(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "dap":
        print("dap seg")
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_dap(x_train, y_train, input_width)
        test_x, test_y = segment_dap(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "pa2":
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_pa2(x_train, y_train, input_width)
        test_x, test_y = segment_pa2(x_test, y_test, input_width)
        print("signal segmented.")
    elif dataset == "sph":
        input_width = 25
        print("segmenting signal...")
        train_x, train_y = segment_sph(x_train, y_train, input_width)
        test_x, test_y = segment_sph(x_test, y_test, input_width)
        print("signal segmented.")
    else:
        print("no correct dataset")

    print("train_x shape =", train_x.shape)
    print("train_y shape =", train_y.shape)
    print("test_x shape =", test_x.shape)
    print("test_y shape =", test_y.shape)

    # http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/

    train = pd.get_dummies(train_y)
    test = pd.get_dummies(test_y)

    train, test = train.align(test, join="inner", axis=1)  # maybe 'outer' is better

    train_y = np.asarray(train)
    test_y = np.asarray(test)

    print("unique test_y", np.unique(test_y))
    print("unique train_y", np.unique(train_y))
    print("test_y[1]=", test_y[1])
    # test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
    print("train_y shape(1-hot) =", train_y.shape)
    print("test_y shape(1-hot) =", test_y.shape)

    # DEFINING THE MODEL
    if dataset == "opp":
        print("opp")
        input_height = 1
        input_width = input_width  # or 90 for actitracker
        num_labels = 18  # or 6 for actitracker
        num_channels = 77  # or 3 for actitracker
    elif dataset == "dap":
        print("dap")
        input_height = 1
        input_width = input_width  # or 90 for actitracker
        num_labels = 2  # or 6 for actitracker
        num_channels = 9  # or 3 for actitracker
    elif dataset == "pa2":
        print("pa2")
        input_height = 1
        input_width = input_width  # or 90 for actitracker
        num_labels = 11  # or 6 for actitracker
        num_channels = 52  # or 3 for actitracker
    elif dataset == "sph":
        print("sph")
        input_height = 1
        input_width = input_width  # or 90 for actitracker
        num_labels = 20  # or 6 for actitracker
        num_channels = 52  # or 3 for actitracker
    else:
        print("wrong dataset")
    batch_size = 64
    stride_size = 2
    kernel_size_1 = 7
    kernel_size_2 = 3
    kernel_size_3 = 1
    depth_1 = 128
    depth_2 = 128
    depth_3 = 128
    num_hidden = 512  # neurons in the fully connected layer

    dropout_1 = tf.placeholder(tf.float32)  # 0.1
    dropout_2 = tf.placeholder(tf.float32)  # 0.25
    dropout_3 = tf.placeholder(tf.float32)  # 0.5

    learning_rate = 0.0005
    training_epochs = 50

    total_batches = train_x.shape[0] // batch_size

    train_x = train_x.reshape(len(train_x), 1, input_width, num_channels)  # opportunity
    test_x = test_x.reshape(len(test_x), 1, input_width, num_channels)  # opportunity
    print("train_x_reshaped = ", train_x.shape)
    print("test_x_reshaped = ", test_x.shape)
    print("train_x shape =", train_x.shape)
    print("train_y shape =", train_y.shape)
    print("test_x shape =", test_x.shape)
    print("test_y shape =", test_y.shape)

    X = tf.placeholder(
        tf.float32, shape=[None, input_height, input_width, num_channels]
    )
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    print("X shape =", X.shape)
    print("Y shape =", Y.shape)

    # HIDDEN LAYERS AND FULLY CONNECTED FOR Opportunity etc
    # https://www.tensorflow.org/get_started/mnist/pros

    # hidden layer 1
    W_conv1 = weight_variable([1, kernel_size_1, num_channels, depth_1])
    b_conv1 = bias_variable([depth_1])

    h_conv1 = tf.nn.relu(depth_conv2d(X, W_conv1) + b_conv1)
    # h_conv1 = tf.nn.dropout(tf.identity(h_conv1), dropout_1)
    h_conv1 = tf.nn.dropout(h_conv1, dropout_1)

    h_pool1 = max_pool(h_conv1, kernel_size_1, stride_size)

    # hidden layer 2
    W_conv2 = weight_variable([1, kernel_size_2, depth_1, depth_2])
    b_conv2 = bias_variable([depth_2])

    h_conv2 = tf.nn.relu(depth_conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, dropout_2)

    h_pool2 = max_pool(h_conv2, kernel_size_2, stride_size)

    # first we get the shape of the last layer and flatten it out
    shape = h_pool2.get_shape().as_list()

    W_fc1 = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
    b_fc1 = bias_variable([num_hidden])

    h_pool3_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, dropout_3)

    # readout layer.

    W_fc2 = weight_variable([num_hidden, num_labels])
    b_fc2 = bias_variable([num_labels])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # COST FUNCTION
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TRAINING THE MODEL
    config = tf.ConfigProto(device_count={"GPU": 0})

    loss_dict = {
        "Train_Accuracy": [],
        "Train_Loss": [],
        "Test_Accuracy": [],
    }
    label_counting_dict = {}
    with tf.Session(config=config) as session:

        tf.compat.v1.initialize_all_variables().run()

        for epoch in range(1):
            print("ONLY ONCE")
            cost_history = np.empty(shape=[0], dtype=float)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                batch_x = train_x[offset : (offset + batch_size), :, :, :]
                batch_y = train_y[offset : (offset + batch_size), :]

                for label in batch_y:
                    if not np.asarray(label).sum() == 1:
                        label_str = "None"
                    else:
                        label_str = f"{np.where(label==1)[0][0]:03d}"

                    if label_str not in label_counting_dict:
                        label_counting_dict[label_str] = 0
                    label_counting_dict[label_str] = label_counting_dict[label_str] + 1

                # print(label_counting_dict)

    return label_counting_dict


def main(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", choices=["opp", "dap", "pa2"], default="opp")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "datasets", "har_dataset"),
    )
    args = parser.parse_args()

    cnn_execute(args.dataset, args.data_path, None)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
