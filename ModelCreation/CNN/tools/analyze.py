import json
import os
import matplotlib.pyplot as plt
# Method -> Percentage -> scores

def analyze(path):
    functions = os.listdir(path)
    max_f_score_of_all_func = 0


    for folder in functions:
        folder_path = os.path.join(path,folder)
        prob_folders = os.listdir(folder_path)

        weighted_f1_score = {}
        for prob_folder in prob_folders:
            score_path = os.path.join(folder_path, prob_folder, "scores.json")
            with open(score_path, "r") as file:
                scores_dict = json.load(file)
                weighted_f1_score[prob_folder[-2:]] = scores_dict["f1_score_w"]

        plt.bar(x=weighted_f1_score)
        plt.show()



if __name__ == "__main__":
    analyze("C:/Dev/Smart_Data/E3")