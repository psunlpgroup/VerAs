from datasets import QuestionSentenceDataset
from metrics import get_eval_metrics
import numpy as np
import pandas as pd
from grader.model import Grader
from grader.tokenization import Tokenizer as GraderTokenizer
from verifier.model import Verifier
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 

def _get_metrics(predictions_p, ground_truths_p, scalers_p):
    ground_truths = torch.from_numpy(np.array(ground_truths_p))
    predictions = torch.from_numpy(np.array(predictions_p))
    scalers = torch.from_numpy(np.array(scalers_p))

    return get_eval_metrics(predictions, ground_truths, scalers)

def get_all_metrics(data_loader, reader, retriever, reader_tokenizer, device, topK, dump_pred=False, dump_file=""):
    with torch.no_grad():
        student_scores = {}
        predictions = []
        ground_truths = []
        test_q = []
        test_report = []
        test_labels = []
        test_whole = []
        test_predictions = []
        test_ids= []
        ver_preds = []
        ver_ground_truths = []
        for _, data in tqdm(enumerate(data_loader)):
            questions, reports, labels, report_IDs, class_weights_for_verifier = data
            labels = labels.to(device)
            class_weights_for_verifier = class_weights_for_verifier.to(device)
           
            verifier_logits, topK_sentences, _ = retriever.verify(questions, reports)
            if verifier_logits.shape[0] ==1:
                verifier_logits = verifier_logits.view(1)
            else:
                verifier_logits = verifier_logits.squeeze()
            verifier_nonzeros = np.array([1 if prob>=0.5 else 0 for prob in verifier_logits.detach().cpu().numpy()]).nonzero()[0]
            test_q.extend(questions)
            test_report.extend(topK_sentences)
            test_whole.extend(reports)
            test_labels.extend(labels.detach().cpu().numpy())
            test_ids.extend(report_IDs)
            ver_preds.extend([1 if prob>=0.5 else 0 for prob in verifier_logits.detach().cpu().numpy()])
            ver_ground_truths.extend([1 if label>0 else 0 for label in labels.cpu().numpy()])

            if len(verifier_nonzeros) != 0:#there is data to pass to the grader
                grader_questions, grader_reports, grader_topK_sentences = np.array(questions).take(verifier_nonzeros), np.array(reports).take(verifier_nonzeros), np.array(topK_sentences).take(verifier_nonzeros)
                questions_tokenized, reports_tokenzied, topK_tokenized = reader_tokenizer(list(grader_questions), list(grader_reports), list(grader_topK_sentences))
                questions_tokenized = questions_tokenized.to(device)
                reports_tokenzied = reports_tokenzied.to(device)
                topK_tokenized = topK_tokenized.to(device)
                logits = reader(questions_tokenized, reports_tokenzied, topK_tokenized)
            
                predictions.extend(logits.cpu()) # contains probability predictions for val loss for 1-5 so for grader
                ground_truths.extend((torch.index_select(labels, 0, torch.from_numpy(verifier_nonzeros).to(device))).cpu()) # TODO:this is not exactly true let me think about this. contains labels 1,2,...5 but with 1 minus versions for grader loss
                pred_probab = torch.nn.Softmax(dim=1)(logits)
                y_pred = pred_probab.argmax(1).cpu() # similar to ground_truths
            else:
                y_pred = []

            all_preds = [0] * labels.shape[0]
            for index, pred in zip(verifier_nonzeros, y_pred):
                all_preds[index] = pred.item() #since grader predicts for 1-5 but its predictions are 0-4
            
            test_predictions.extend(all_preds)
            for id,pred,gt in zip(report_IDs, all_preds, labels.cpu()):
                act_gt = gt.item()
                act_pred = pred 
                
                if id in student_scores:
                    cum_preds, cum_gts = student_scores[id]
                    student_scores[id] = (cum_preds + act_pred, cum_gts + act_gt)
                else:
                    student_scores[id] = (act_pred, act_gt)
        
        predictions = torch.tensor([pred for pred,_ in student_scores.values()])
        ground_truths = torch.tensor([gt for _,gt in student_scores.values()])
        metrics = _get_metrics(predictions, ground_truths, torch.ones_like(predictions))
        if dump_pred:
            pd.DataFrame({"question":test_q, "filtered_report":test_report, "whole_report":test_whole, "labels":test_labels, "preds":test_predictions, "IDs":test_ids,"ver_preds":ver_preds}).to_csv(dump_file)
    return metrics
    

grader_file_path = sys.argv[1]
verifier_file_path = sys.argv[2]


config = {
    "test_folder": "data/test",
    "val_folder": "data/val",
    "test_labels": "data/labels/test.csv",
    "val_labels": "data/labels/val.csv",
    "batch_size": 4,
    "rubric_dimension":"data/rubric_dimensions.json",
}  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
test_set = QuestionSentenceDataset(config["test_folder"], config["test_labels"], 
        config["rubric_dimension"], is_val=True)
test_loader = DataLoader(test_set, 1)

verifier = Verifier(bert_or_sbert="sbert", topK=3)
verifier.load_state_dict(torch.load(f"{verifier_file_path}/model.pth",map_location=torch.device('cpu')))
verifier = verifier.to(device)
verifier.eval()

grader = Grader("electra")
grader.load_state_dict(torch.load(f"{grader_file_path}/model.pth",map_location=torch.device('cpu')))
grader = grader.to(device)
grader.eval()
grader_tokenizer = GraderTokenizer("electra")


test_metrics = get_all_metrics(test_loader, grader, verifier, grader_tokenizer, device, 3, True, "test_results.csv")

print("\ntest:", end=" ")
for metric, value in test_metrics.items():
    print(f"{metric}: {round(value,3)},",end=" ")
print()
