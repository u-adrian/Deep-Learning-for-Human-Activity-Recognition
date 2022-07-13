import json
import os
import matplotlib.pyplot as plt

# Method -> Percentage -> scores
import numpy as np

# from ModelCreation.CNN.cnn1d import label_counting


def listdir_nohidden(path):
    dirs = []
    for f in os.listdir(path):
        if not f.startswith("."):
            dirs.append(f)
    return dirs


def analyze(technique, path, range, f1_score, save_path):
    functions = os.listdir(path)

    dictionary = {}

    function_nums = np.arange(len(functions)) + 1

    for ffolder in functions:
        folder_path = os.path.join(path, ffolder)
        hyper_folders = listdir_nohidden(folder_path)

        for hfolder in hyper_folders:
            if hfolder not in dictionary.keys():
                dictionary[hfolder] = {"mean": [], "std": []}

            hyper_folder_path = os.path.join(folder_path, hfolder)

            cv_folders = listdir_nohidden(hyper_folder_path)
            cv_scores = np.zeros(len(cv_folders))

            for i, cv_folder in enumerate(cv_folders):
                cv_folder_path = os.path.join(hyper_folder_path, cv_folder)

                score_path = os.path.join(cv_folder_path, "scores.json")
                with open(score_path, "r") as file:
                    scores_dict = json.load(file)
                    cv_scores[i] = scores_dict[f1_score][0]

            dictionary[hfolder]["mean"].append(np.mean(cv_scores))
            dictionary[hfolder]["std"].append(np.std(cv_scores))

    plt.ylim(range[0], range[1])

    if technique == "supervised":
        # Baseline
        plt.bar(
            function_nums[0],
            dictionary["test_50"]["mean"][0],
            yerr=dictionary["test_50"]["std"][0],
            width=0.2,
            color="grey",
            edgecolor="black",
            align="center",
        )

        # augmentations
        plt.bar(
            function_nums[1:] - 0.2,
            dictionary["test_20"]["mean"][1:],
            yerr=dictionary["test_20"]["std"][1:],
            width=0.2,
            color="b",
            edgecolor="black",
            align="center",
            label="0.20",
        )
        plt.bar(
            function_nums[1:],
            dictionary["test_50"]["mean"][1:],
            yerr=dictionary["test_50"]["std"][1:],
            width=0.2,
            color="g",
            edgecolor="black",
            align="center",
            label="0.50",
        )
        plt.bar(
            function_nums[1:] + 0.2,
            dictionary["test_80"]["mean"][1:],
            yerr=dictionary["test_80"]["std"][1:],
            width=0.2,
            color="r",
            edgecolor="black",
            align="center",
            label="0.80",
        )
    else:
        # Baseline
        plt.bar(
            function_nums[0],
            dictionary["test_N=1_M=3"]["mean"][0],
            yerr=dictionary["test_N=1_M=3"]["std"][0],
            width=0.2,
            color="grey",
            edgecolor="black",
            align="center",
        )
        # Hyperparameter
        plt.bar(
            function_nums[1:] - 0.3,
            dictionary["test_N=1_M=3"]["mean"][1:],
            yerr=dictionary["test_N=1_M=3"]["std"][1:],
            width=0.2,
            color="b",
            edgecolor="black",
            align="center",
            label="N=1 M=3",
        )
        plt.bar(
            function_nums[1:] - 0.1,
            dictionary["test_N=2_M=3"]["mean"][1:],
            yerr=dictionary["test_N=2_M=3"]["std"][1:],
            width=0.2,
            color="g",
            edgecolor="black",
            align="center",
            label="N=2 M=3",
        )
        plt.bar(
            function_nums[1:] + 0.1,
            dictionary["test_N=3_M=3"]["mean"][1:],
            yerr=dictionary["test_N=3_M=3"]["std"][1:],
            width=0.2,
            color="r",
            edgecolor="black",
            align="center",
            label="N=3 M=3",
        )
        plt.bar(
            function_nums[1:] + 0.3,
            dictionary["test_N=4_M=3"]["mean"][1:],
            yerr=dictionary["test_N=4_M=3"]["std"][1:],
            width=0.2,
            color="y",
            edgecolor="black",
            align="center",
            label="N=4 M=3",
        )

    plt.xticks(function_nums, functions, fontsize=9)

    plt.ylabel(f1_score)
    plt.legend()
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def analyze_supervised():
    path = "/home/kit/stud/ucxam/output/human_activity"
    save_path = "/home/kit/stud/ucxam/output/visualizations/"

    analyze(
        "supervised",
        os.path.join(path, "opp"),
        range=(0.4, 0.95),
        f1_score="f1_score_m",
        save_path=os.path.join(save_path, "opp_f1_score_m.jpg"),
    )
    analyze(
        "supervised",
        os.path.join(path, "dap"),
        range=(0.4, 0.8),
        f1_score="f1_score_m",
        save_path=os.path.join(save_path, "dap_f1_score_m.jpg"),
    )
    analyze(
        "supervised",
        os.path.join(path, "pa2"),
        range=(0.4, 0.9),
        f1_score="f1_score_m",
        save_path=os.path.join(save_path, "pa2_f1_score_m.jpg"),
    )

    analyze(
        "supervised",
        os.path.join(path, "opp"),
        range=(0.8, 0.95),
        f1_score="f1_score_w",
        save_path=os.path.join(save_path, "opp_f1_score_w.jpg"),
    )
    analyze(
        "supervised",
        os.path.join(path, "pa2"),
        range=(0.4, 0.9),
        f1_score="f1_score_w",
        save_path=os.path.join(save_path, "pa2_f1_score_w.jpg"),
    )

    # plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="opp")
    # plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="pa2")
    # plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="dap")


def analyze_unsupervised():
    path = os.path.join(os.path.expanduser("~"), "output", "human_activity")
    save_path = os.path.join(os.path.expanduser("~"), "output", "visualizations")

    analyze(
        "unsupervised",
        os.path.join(path, "pa2"),
        range=(0.0, 1.0),
        f1_score="f1_score_m",
        save_path=os.path.join(save_path, "pa2_f1_score_m.jpg"),
    )
    analyze(
        "unsupervised",
        os.path.join(path, "pa2"),
        range=(0.0, 1.0),
        f1_score="f1_score_w",
        save_path=os.path.join(save_path, "pa2_f1_score_w.jpg"),
    )


def plot_label_hist(dataset_path, dataset_name):
    label_dict = label_counting(dataset_name, dataset_path)
    lists = sorted(label_dict.items())
    x, y = zip(*lists)

    print(label_dict)
    y_max = np.max(np.asarray(y) * 1.05)
    plt.ylim(0, y_max)
    plt.bar(x, y, width=0.2, color="b", align="center")
    plt.xticks(rotation=-45)
    plt.ylabel("# samples in one epoch")

    plt.show()


if __name__ == "__main__":
    # analyze_supervised()
    analyze_unsupervised()
