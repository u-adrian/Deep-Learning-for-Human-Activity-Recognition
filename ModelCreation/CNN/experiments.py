import argparse
import json
import os.path

import numpy as np

from cnn1d import cnn_execute
from sup_augmentations import (
    aug_noise,
    aug_convolve,
    aug_crop,
    aug_drift,
    aug_dropout,
    aug_pool,
    aug_quantize,
)
from unsup_augmentations import rand_aug

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def hyper_param_eval(
    supervised, aug_func, name, output_path, data_path, dataset_name, seed
):
    output_path_experiment = os.path.join(output_path, name)

    if supervised:
        prob_list = [0.2, 0.5, 0.8]

        for i, prob in enumerate(prob_list):
            aug_function = None
            if aug_func is not None:

                def aug_function(b_x, b_y):
                    return aug_func(b_x, b_y, prob)

            np.random.seed(seed)
            tf.set_random_seed(seed)

            loss_dicts, score_dicts, confusion_matrices = cnn_execute(
                dataset_name, data_path=data_path, aug_function=aug_function
            )
            output_path_prob = os.path.join(
                output_path_experiment, f"test_{int(prob*100):02d}"
            )

            os.makedirs(output_path_prob, exist_ok=True)

            write_training_outcomes(
                loss_dicts, score_dicts, confusion_matrices, output_path_prob
            )
    else:
        n_m_combinations = [(1, 3), (2, 3), (3, 3), (4, 3)]

        for i, comb in enumerate(n_m_combinations):
            aug_function = None
            if aug_func is not None:

                def aug_function(b_x, b_y):
                    return aug_func(b_x, b_y, *comb)

            np.random.seed(seed)
            tf.set_random_seed(seed)

            loss_dicts, score_dicts, confusion_matrices = cnn_execute(
                dataset_name, data_path=data_path, aug_function=aug_function
            )
            output_path_hyper = os.path.join(
                output_path_experiment, f"test_N={comb[0]}_M={comb[1]}"
            )

            os.makedirs(output_path_hyper, exist_ok=True)

            write_training_outcomes(
                loss_dicts, score_dicts, confusion_matrices, output_path_hyper
            )


def write_training_outcomes(
    loss_dicts, score_dicts, confusion_matrices, output_path_hyper
):
    for i, (loss_dict, score_dict, confusion_matrix) in enumerate(
        zip(loss_dicts, score_dicts, confusion_matrices)
    ):
        output_path_cv = os.path.join(output_path_hyper, f"cv_{i:02d}")

        os.makedirs(output_path_cv, exist_ok=True)

        output_path_loss = os.path.join(output_path_cv, "losses.json")
        output_path_score = os.path.join(output_path_cv, "scores.json")
        output_path_matrix = os.path.join(output_path_cv, "confusion_matrix.npy")

        with open(output_path_loss, "w") as file:
            json.dump(loss_dict, file)

        with open(output_path_score, "w") as file:
            json.dump(score_dict, file)

        with open(output_path_matrix, "wb") as file:
            np.save(file, confusion_matrix)


def main(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--technique", choices=["supervised", "unsupervised"], default="supervised"
    )
    parser.add_argument("--dataset", choices=["opp", "dap", "pa2"], default="opp")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "datasets", "har_dataset"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "output", "human_activity"),
    )
    args = parser.parse_args()

    technique = args.technique
    output_path = args.output_path
    data_path = args.data_path
    dataset_name = args.dataset

    if technique == "supervised":
        supervised_experiment(output_path, data_path, dataset_name)
    else:
        unsupervised_experiment(output_path, data_path, dataset_name)


def supervised_experiment(output_path, data_path, dataset_name):
    seed = 1

    augmentation_dict = {
        "baseline": None,
        "noise": aug_noise,
        "convolve": aug_convolve,
        "crop": aug_crop,
        "drift": aug_drift,
        "dropout": aug_dropout,
        "pool": aug_pool,
        "quantize": aug_quantize,
    }

    for augmentation_name in augmentation_dict:
        print(f"Starting hyperparameter search for '{augmentation_name}'")
        augmentation_func = augmentation_dict[augmentation_name]
        hyper_param_eval(
            supervised=True,
            aug_func=augmentation_func,
            name=augmentation_name,
            output_path=output_path,
            data_path=data_path,
            dataset_name=dataset_name,
            seed=seed,
        )


def unsupervised_experiment(output_path, data_path, dataset_name):
    seed = 42

    method_dict = {
        "baseline": None,
        "RandAug": rand_aug,
    }

    for aug_name, func in method_dict.items():
        print(f"Starting hyperparameter search for RandAug")
        augmentation_func = func
        hyper_param_eval(
            supervised=False,
            aug_func=augmentation_func,
            name=aug_name,
            output_path=output_path,
            data_path=data_path,
            dataset_name=dataset_name,
            seed=seed,
        )


if __name__ == "__main__":
    main(argparse.ArgumentParser())
