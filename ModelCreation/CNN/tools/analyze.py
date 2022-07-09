import json
import os
import matplotlib.pyplot as plt
# Method -> Percentage -> scores

def analyze(path):
    functions = os.listdir(path)
    max_f_score_of_all_func = 0

    dictionary = {}

    function_nums = list(range(len(functions)))

    for folder in functions:
        folder_path = os.path.join(path,folder)
        prob_folders = os.listdir(folder_path)

        for prob_folder in prob_folders:
            if prob_folder not in dictionary.keys():
                dictionary[prob_folder] = []

            score_path = os.path.join(folder_path, prob_folder, "scores.json")
            with open(score_path, "r") as file:
                scores_dict = json.load(file)
                dictionary[prob_folder].append(scores_dict["f1_score_w"])

    plt.bar(function_nums - 0.2, dictionary["test_20"], width=0.2, color="b", align="center")
    plt.bar(function_nums, dictionary["test_50"], width=0.2, color="g", align="center")
    plt.bar(function_nums + 0.2, dictionary["test_80"], width=0.2, color="r", align="center")
    plt.show()



if __name__ == "__main__":
    analyze("C:/Dev/Smart_Data/E3")