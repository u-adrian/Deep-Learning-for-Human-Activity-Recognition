import argparse
import json
import os.path

import numpy as np

from cnn1d import cnn_execute
from sup_augmentations import aug_noise, aug_convolve, aug_crop, aug_drift, aug_dropout, aug_pool, \
    aug_quantize


def hyper_param_eval(aug_func, name, output_path, data_path, dataset_name):
    output_path_experiment = os.path.join(output_path, name)
    prob_list = [0.2, 0.5, 0.8]

    for i, prob in enumerate(prob_list):
        def aug_function(b_x, b_y):
            return aug_func(b_x, b_y, prob)

        loss_dict, score_dict, confusion_matrix = cnn_execute(dataset_name, data_path=data_path, aug_function=aug_function)
        output_path_prob = os.path.join(output_path_experiment, f"test_{int(prob*100):02d}")

        os.makedirs(output_path_prob, exist_ok=True)

        output_path_loss = os.path.join(output_path_prob, "losses.json")
        output_path_score = os.path.join(output_path_prob, "scores.json")
        output_path_matrix = os.path.join(output_path_prob, "confusion_matrix.json")

        with open(output_path_loss, "w") as file:
            json.dump(loss_dict, file)

        with open(output_path_score, "w") as file:
            json.dump(score_dict, file)

        with open(output_path_matrix, "wb") as file:
            np.save(file, confusion_matrix)


def main(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", choices=["opp", "dap", "pa2"], default="opp")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(os.path.expanduser("~"), "datasets", "har_dataset"))
    parser.add_argument("--output_path", type=str,
                        default=os.path.join(os.path.expanduser("~"), "output", "human_activity"))
    args = parser.parse_args()

    output_path = args.output_path
    data_path = args.data_path
    dataset_name = args.dataset

    augmentation_dict = {
        "noise": aug_noise,
        "convolve": aug_convolve,
        "crop": aug_crop,
        "drift": aug_drift,
        "dropout": aug_dropout,
        "pool": aug_pool,
        "quantize": aug_quantize
    }

    for augmentation_name in augmentation_dict:
        print(f"Starting hyperparameter search for '{augmentation_name}'")
        augmentation_func = augmentation_dict[augmentation_name]
        hyper_param_eval(aug_func=augmentation_func, name=augmentation_name, output_path=output_path,
                         data_path=data_path, dataset_name=dataset_name)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
