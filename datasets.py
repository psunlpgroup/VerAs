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
       
        self.labels = _add_rubric_defs(self.labels, self.rubric)
        self.labels = _create_combination_data(self.labels)
        #if not is_val:
        #    self.labels = pd.concat([self.labels, self.labels], ignore_index=True)# augmentation by doubling the data
        self.is_val = is_val
        temp = [0 if y==0 else 1 for y in self.labels["Score"]]
        self.w = len(temp) / (2 * np.bincount(temp))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        row = self.labels.iloc[index]
        verifier_label = 1 if row['Score']>0 else 0
        label = row['Score']
        with open(f'{self.data_folder}/{row["ID"]}.txt') as f:
            report = f.read()
        if self.is_val:
            return row["Question"], report, torch.tensor(label, dtype=torch.long), row["ID"], torch.tensor(self.w[verifier_label], dtype=torch.float)
        else:
            return row["Question"], report, torch.tensor(label, dtype=torch.long), torch.tensor(self.w[verifier_label], dtype=torch.float)
