import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def _add_rubric_defs(df, rubric):
    lab1_rubric = rubric["1"]
    lab2_rubric = rubric["2"]
    for i in range(1,8):
        df[f"Question {i}"] = ""
        df.loc[df["LabNo"] == 1, f"Question {i}"] = lab1_rubric[str(i)]
        df.loc[df["LabNo"] == 2, f"Question {i}"] = lab2_rubric[str(i)]

    df.loc[df["LabNo"] == 1, f"Question 8"] = lab1_rubric[str(8)]
    return df

def _create_combination_data(df):
    lab1_df = df[df["LabNo"] == 1]
    lab2_df = df[df["LabNo"] == 2]

    lab1_final = lab1_df[["Dimension 1", "ID", "LabNo", "Question 1"]]
    lab1_final.rename(columns={"Dimension 1": "Score", "Question 1": "Question"}, inplace=True)
    lab2_final = lab2_df[["Dimension 1", "ID", "LabNo", "Question 1"]]
    lab2_final.rename(columns={"Dimension 1": "Score", "Question 1": "Question"}, inplace=True)

    for i in range(2,8):
        temp = lab1_df[[f"Dimension {i}", "ID", "LabNo", f"Question {i}"]]
        temp.rename(columns={f"Dimension {i}": "Score", f"Question {i}": "Question"}, inplace=True)
        lab1_final = pd.concat([lab1_final, temp], ignore_index=True)

        temp = lab2_df[[f"Dimension {i}", "ID", "LabNo", f"Question {i}"]]
        temp.rename(columns={f"Dimension {i}": "Score", f"Question {i}": "Question"}, inplace=True)
        lab2_final = pd.concat([lab2_final, temp], ignore_index=True)

    temp = lab1_df[[f"Dimension 8", "ID", "LabNo", f"Question 8"]]
    temp.rename(columns={f"Dimension 8": "Score", f"Question 8": "Question"}, inplace=True)
    lab1_final = pd.concat([lab1_final, temp], ignore_index=True)

    return pd.concat([lab1_final, lab2_final], ignore_index=True)

"""This is the dataset that I use for my approach. It provides data for the retriever then retriever will pass the topK to the reader."""
class QuestionSentenceDataset(Dataset):
    def __init__(self, 
        data_folder: str, 
        labels_file: str, 
        rubric_dimensions_file: str,
        max_sentence_length: int=122,
        dataset_name: str="college_physics", # or it can be middle_school
        is_val: bool=False):

        super().__init__()
        self.data_folder = data_folder
        self.max_sentence_length = max_sentence_length
        self.labels = pd.read_csv(labels_file)
        with open(rubric_dimensions_file,"r") as f:
            self.rubric = json.load(f)
        self.score_dict = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five"
        }
        
        if dataset_name == "middle_school":
            self.identifiers = {
                "1O1": "1_2",
                "1O0": "4",
                "1R1": "2",
                "1R0": "4",
                "2O1": "3",
                "2O0": "5",
                "2R0": "5",
                "2R1": "3"
            }
            essay1_labels = self.labels[self.labels["essay_number"]==1]

            rubric = self.rubric["1"]
            for i in range(1,7):
                essay1_labels[f"Question {i}"] = rubric[str(i)]
            
            final = essay1_labels[["Dimension 1", "ID", "Question 1", "essay_number", "essay_version", "ground_truth"]]
            final.rename(columns={"Dimension 1": "Score", "Question 1": "Question"}, inplace=True)
            for i in range(2,7):
                temp = essay1_labels[[f"Dimension {i}", "ID", f"Question {i}", "essay_number", "essay_version", "ground_truth"]]
                temp.rename(columns={f"Dimension {i}": "Score", f"Question {i}": "Question"}, inplace=True)
                final = pd.concat([final, temp], ignore_index=True)
            essay1_labels = final
            # second essay
            essay2_labels = self.labels[self.labels["essay_number"]==2]

            rubric = self.rubric["2"]
            for i in range(1,9):
                essay2_labels[f"Question {i}"] = rubric[str(i)]
            
            final = essay2_labels[["Dimension 1", "ID", "Question 1", "essay_number", "essay_version", "ground_truth"]]
            final.rename(columns={"Dimension 1": "Score", "Question 1": "Question"}, inplace=True)
            for i in range(2,7):
                temp = essay2_labels[[f"Dimension {i}", "ID", f"Question {i}", "essay_number", "essay_version", "ground_truth"]]
                temp.rename(columns={f"Dimension {i}": "Score", f"Question {i}": "Question"}, inplace=True)
                final = pd.concat([final, temp], ignore_index=True)
            self.labels = pd.concat([final, essay1_labels], ignore_index=True)
        else: # college physics
            self.labels = _add_rubric_defs(self.labels, self.rubric)
            self.labels = _create_combination_data(self.labels)

        self.is_val = is_val
        temp = [0 if y==0 else 1 for y in self.labels["Score"]]
        self.w = len(temp) / (2 * np.bincount(temp))
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        row = self.labels.iloc[index]
        verifier_label = 1 if row['Score']>0 else 0
        label = row['Score']
        if self.dataset_name == "college_physics":
            with open(f'{self.data_folder}/{row["ID"]}.txt') as f:
                report = f.read()
        else:
            data_section = self.identifiers[self.create_identifier(row)]
            try:
                with open(f'{self.data_folder}/{data_section}/{row["ID"]}.txt') as f:
                    report = f.read()
            except:
                with open(f'{self.data_folder}/{data_section}/{row["ID"]}') as f:
                    report = f.read()
        if self.is_val:
            return row["Question"], report, torch.tensor(label, dtype=torch.long), row["ID"], torch.tensor(self.w[verifier_label], dtype=torch.float)
        else:
            return row["Question"], report, torch.tensor(label, dtype=torch.long), torch.tensor(self.w[verifier_label], dtype=torch.float)

    def create_identifier(self, row):
        return str(int(row["essay_number"])) + str(row["essay_version"]) + str(int(row["ground_truth"]))