import json
import os
import matplotlib.pyplot as plt
# Method -> Percentage -> scores
import numpy as np


def analyze(path):
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
                dictionary[prob_folder].append(scores_dict["f1_score_w"][0])

    plt.ylim(0.6,0.9)
    plt.bar(function_nums - 0.2, dictionary["test_20"], width=0.2, color="b", align="center", label="0.20")
    plt.bar(function_nums, dictionary["test_50"], width=0.2, color="g", align="center", label="0.50")
    plt.bar(function_nums + 0.2, dictionary["test_80"], width=0.2, color="r", align="center", label="0.80")
    plt.xticks(function_nums, functions)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    analyze("C:/Dev/Smart_Data/Ergebnisse/pa2/pa2")