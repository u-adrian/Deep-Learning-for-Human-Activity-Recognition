import json
import os
import matplotlib.pyplot as plt
# Method -> Percentage -> scores
import numpy as np

from ModelCreation.CNN.cnn1d import label_counting


def analyze(path, range, f1_score):
    functions = os.listdir(path)
    max_f_score_of_all_func = 0

    dictionary = {}

    function_nums = np.arange(len(functions))+1

    for folder in functions:
        folder_path = os.path.join(path,folder)
        prob_folders = os.listdir(folder_path)

        for prob_folder in prob_folders:
            if prob_folder not in dictionary.keys():
                dictionary[prob_folder] = []

            score_path = os.path.join(folder_path, prob_folder, "scores.json")
            with open(score_path, "r") as file:
                scores_dict = json.load(file)
                dictionary[prob_folder].append(scores_dict[f1_score][0])

    plt.ylim(range[0], range[1])
    #Baseline
    plt.bar(function_nums[0] - 0.2, dictionary["test_20"][0], width=0.2, color="grey", edgecolor="black",align="center")
    plt.bar(function_nums[0], dictionary["test_50"][0], width=0.2, color="grey", edgecolor="black", align="center")
    plt.bar(function_nums[0] + 0.2, dictionary["test_80"][0], width=0.2, color="grey", edgecolor="black", align="center")

    #augmentations
    plt.bar(function_nums[1:] - 0.2, dictionary["test_20"][1:], width=0.2, color="b", edgecolor="black", align="center", label="0.20")
    plt.bar(function_nums[1:], dictionary["test_50"][1:], width=0.2, color="g", edgecolor="black", align="center", label="0.50")
    plt.bar(function_nums[1:] + 0.2, dictionary["test_80"][1:], width=0.2, color="r", edgecolor="black", align="center", label="0.80")

    plt.xticks(function_nums, functions, fontsize=9)

    plt.ylabel(f1_score)
    plt.legend()
    plt.show()


def plot_label_hist(dataset_path, dataset_name):
    label_dict = label_counting(dataset_name, dataset_path)
    lists = sorted(label_dict.items())
    x, y = zip(*lists)

    print(label_dict)
    y_max = np.max(np.asarray(y)*1.05)
    plt.ylim(0,y_max)
    plt.bar(x,y, width=0.2, color="b", align="center")
    plt.xticks(rotation=-45)
    plt.ylabel("# samples in one epoch")

    plt.show()



if __name__ == "__main__":
    analyze("C:/Dev/Smart_Data/Ergebnisse/human_activity/human_activity/opp", range=(0.4,0.95), f1_score="f1_score_m")
    analyze("C:/Dev/Smart_Data/Ergebnisse/human_activity/human_activity/dap", range=(0.4,0.8), f1_score="f1_score_m")
    analyze("C:/Dev/Smart_Data/Ergebnisse/human_activity/human_activity/pa2", range=(0.4, 0.9), f1_score="f1_score_m")

    analyze("C:/Dev/Smart_Data/Ergebnisse/human_activity/human_activity/opp", range=(0.8, 0.95), f1_score="f1_score_w")
    analyze("C:/Dev/Smart_Data/Ergebnisse/human_activity/human_activity/pa2", range=(0.4, 0.9), f1_score="f1_score_w")

    #plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="opp")
    #plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="pa2")
    #plot_label_hist(dataset_path="C:/Users/adria/Downloads/", dataset_name="dap")
