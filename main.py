import argparse
from copy import deepcopy
from datasets import QuestionSentenceDataset
import loss_functions
import numpy as np
from metrics import spearman, acc,get_eval_metrics
import os
from grader.model import Grader
from grader.tokenization import Tokenizer as GraderTokenizer
from verifier.model import Verifier
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from tqdm import tqdm
import utils 


parser = argparse.ArgumentParser()
parser.add_argument("--top_k", default=3)
parser.add_argument("--dataset_name", default="college_physics", choices=["college_physics", "middle_school"])
parser.add_argument("--verifier_model", default="sbert", choices=["bert", "sbert"])
parser.add_argument("--loss_function", default="oll", choices=["oll", "cross_entropy"])
parser.add_argument("--grader_model", default="electra", choices=["longt5", "bert", "electra"])
parser.add_argument("--oll_loss_alpha", default=2.5)

args = parser.parse_args()

loss_dict = {
    "oll": loss_functions.oll_loss,
    "cross_entropy": loss_functions.cross_entropy_loss if args.dataset_name == "college_physics" else torch.nn.functional.binary_cross_entropy
}

if args.dataset_name == "college_physics":
    train_folder = "data/train"
    val_folder = "data/val"
    train_labels = "data/labels/train.csv"
    val_labels = "data/labels/val.csv"
    rubric_dimension = "data/rubric_dimensions.json"
else:
    train_folder = "middle_school_data"
    val_folder = "middle_school_data"
    train_labels = "middle_school_data/middle_school_essay1_2_train.csv"
    val_labels = "middle_school_data/middle_school_essay1_2_val.csv"
    rubric_dimension = "middle_school_data/rubric_dimensions.json"

config = {
    "train_folder": train_folder,
    "val_folder": val_folder,
    "train_labels": train_labels,
    "val_labels": val_labels,
    "rubric_dimension": rubric_dimension,
    "batch_size": 4,
    "epoch": 8,
    "lr": 0.00005,
}  

loss_function_name = f"oll{args.oll_loss_alpha}" if args.loss_function == "oll" else args.loss_function

# original lr  [0.00005, 0.0005, 0.005]
for lr in [0.00005, 0.00001]:
    for batch_size in [4,8]:
        utils.set_all_seeds(99) 
        config["lr"] = lr
        config["batch_size"] = batch_size
        
        model_name = f"{loss_function_name}-{args.grader_model}-{args.verifier_model}-{lr}-{batch_size}"
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_set = QuestionSentenceDataset(config["train_folder"], config["train_labels"], 
                config["rubric_dimension"], dataset_name=args.dataset_name)
        train_loader = DataLoader(train_set, 1, shuffle=True)

        val_set = QuestionSentenceDataset(config["val_folder"], config["val_labels"], 
            config["rubric_dimension"], dataset_name=args.dataset_name, is_val=True)
        val_loader = DataLoader(val_set,1)

        verifier = Verifier(bert_or_sbert=args.verifier_model, topK=int(args.top_k))
        verifier = verifier.to(device)
        verifier.train()

        grader = Grader(args.grader_model, args.dataset_name)
        
        optimizer = Adam(list(grader.parameters()), lr=config["lr"])
        verifier_optimizer = Adam(list(verifier.parameters()), lr=0.00001)
        
        grader_tokenizer = GraderTokenizer(args.grader_model)
        criterion = loss_dict[args.loss_function]
        verifier_criterion = torch.nn.functional.binary_cross_entropy
        
        best_grader = None
        best_verifier = None
        best_epoch = -1
        best_spearman = -2
        best_val_loss_grader = 100000
        best_val_loss_verifier = 100000
        
        print("initial verifier training")
        predictions = []
        ground_truths = []
        for e in tqdm(range(1)):
            predictions = []
            ground_truths = []
            update_params = 0 # this is for imitating batch size
            for i, data in enumerate(train_loader):
                questions, reports, labels, class_weights_for_verifier = data
                labels = labels.to(device)
                class_weights_for_verifier = class_weights_for_verifier.to(device)
                verifier_labels_lst = [1 if label>0 else 0 for label in labels.cpu().numpy()]
                verifier_labels = torch.tensor(verifier_labels_lst, dtype=torch.float).to(device)
                
                verifier_logits, _,_ = verifier.verify(questions, reports)

                if verifier_logits.shape[0] ==1:
                        verifier_logits = verifier_logits.view(1)
                else:
                    verifier_logits = verifier_logits.squeeze()
                verifier_loss = verifier_criterion(verifier_logits, verifier_labels, class_weights_for_verifier)
                predictions.extend([1 if prob>=0.5 else 0 for prob in verifier_logits.detach().cpu().numpy()])
                ground_truths.extend(verifier_labels_lst)
                verifier_loss.backward()
                update_params +=1 # batch size in data loader is 2
                if update_params == config["batch_size"] or i == len(train_loader) -1: # the second condition is for the case where the last batch is not equal to batch size
                    update_params = 0
                    verifier_optimizer.step()
                    verifier_optimizer.zero_grad() 
            print("Train accuracy of verifier after its initial training: ", acc(predictions, ground_truths))
        print("number of 1s and 0s in verifiers predictions:", predictions.count(1), predictions.count(0))
        print("starts regular training")
        grader = grader.to(device)
        for e in tqdm(range(config["epoch"])):
            running_loss = 0
            update_params = 0 # this is for imitating batch size
            for i, data in enumerate(train_loader):
                questions, reports, labels, class_weights_for_verifier = data
                labels = labels.to(device)
                class_weights_for_verifier = class_weights_for_verifier.to(device)
                verifier_labels = torch.tensor([1 if label>0 else 0 for label in labels.cpu().numpy()], dtype=torch.float).to(device)
                
                verifier.train()
                verifier_logits, topK_sentences,_ = verifier.verify(questions, reports)
                if verifier_logits.shape[0] ==1:
                        verifier_logits = verifier_logits.view(1)
                else:
                    verifier_logits = verifier_logits.squeeze()
                verifier_loss = verifier_criterion(verifier_logits, verifier_labels, class_weights_for_verifier)
                
                update_params +=1
                verifier_nonzeros = np.array([1 if prob>=0.5 else 1 for prob in verifier_logits.detach().cpu().numpy()]).nonzero()[0]
                
                if len(verifier_nonzeros) == 0:#there is no data to pass to the grader
                    running_loss += 0 #deepcopy(verifier_loss.item())*len(data[2])
                    verifier_loss.backward()
                    if update_params == config["batch_size"] or i == len(train_loader) -1: # the second condition is for the case where the last batch is not equal to batch size
                        update_params = 0
                        verifier_optimizer.step()
                        verifier_optimizer.zero_grad() 
                    continue

                grader_questions, grader_reports, grader_topK_sentences = np.array(questions).take(verifier_nonzeros), np.array(reports).take(verifier_nonzeros), np.array(topK_sentences).take(verifier_nonzeros)
                
                grader.train()
            
                questions_tokenized, reports_tokenized, topK_tokenized = grader_tokenizer(list(grader_questions), list(grader_reports), list(grader_topK_sentences))
                questions_tokenized = questions_tokenized.to(device)
                reports_tokenized = reports_tokenized.to(device)
                topK_tokenized = topK_tokenized.to(device)

                logits = grader(questions_tokenized, reports_tokenized, topK_tokenized)
                modified_labels = torch.index_select(labels, 0, torch.from_numpy(verifier_nonzeros).to(device)) 
                if args.loss_function == "oll":
                    loss = criterion(logits, modified_labels , alpha = float(args.oll_loss_alpha))
                else:
                    loss = criterion(logits, modified_labels)
                running_loss += deepcopy(loss.item())*len(data[2])
                loss.backward()
                verifier_loss.backward()
                if update_params == config["batch_size"] or i == len(train_loader) -1: # the second condition is for the case where the last batch is not equal to batch size
                    update_params = 0
                    optimizer.step()
                    verifier_optimizer.step()
                    optimizer.zero_grad() 
                    verifier_optimizer.zero_grad()
                
            grader.eval()
            verifier.eval()
            with torch.no_grad():
                running_verifier_loss_val = 0
                student_scores = {}
                predictions = []
                ground_truths = []
                grader_actual_preds = []
                ver_preds = []
                ver_ground_truths = []
                for i, data in tqdm(enumerate(val_loader)):
                    questions, reports, labels, report_IDs, class_weights_for_verifier = data
                    labels = labels.to(device)
                    class_weights_for_verifier = class_weights_for_verifier.to(device)
                    verifier_labels = torch.tensor([1 if label>0 else 0 for label in labels.cpu().numpy()], dtype=torch.float).to(device)

                    verifier_logits, topK_sentences, _ = verifier.verify(questions, reports)
                    if verifier_logits.shape[0] ==1:
                        verifier_logits = verifier_logits.view(1)
                    else:
                        verifier_logits = verifier_logits.squeeze()
                    verifier_loss = verifier_criterion(verifier_logits, verifier_labels, class_weights_for_verifier)
                    verifier_nonzeros = np.array([1 if prob>=0.5 else 0 for prob in verifier_logits.detach().cpu().numpy()]).nonzero()[0]
                    
                    ver_preds.extend([1 if prob>=0.5 else 0 for prob in verifier_logits.detach().cpu().numpy()])
                    ver_ground_truths.extend([1 if label>0 else 0 for label in labels.cpu().numpy()])

                    running_verifier_loss_val += deepcopy(verifier_loss.item())*len(data[2])
                    if len(verifier_nonzeros) != 0:#there is data to pass to the grader
                        grader_questions, grader_reports, grader_topK_sentences = np.array(questions).take(verifier_nonzeros), np.array(reports).take(verifier_nonzeros), np.array(topK_sentences).take(verifier_nonzeros)
                        questions_tokenized, reports_tokenized, topK_tokenized = grader_tokenizer(list(grader_questions), list(grader_reports), list(grader_topK_sentences))
                        questions_tokenized = questions_tokenized.to(device)
                        reports_tokenized = reports_tokenized.to(device)
                        topK_tokenized = topK_tokenized.to(device)
                        logits = grader(questions_tokenized, reports_tokenized, topK_tokenized)
                    
                        predictions.extend(logits.cpu()) # contains probability predictions for val loss for 1-5 so for grader
                        ground_truths.extend((torch.index_select(labels, 0, torch.from_numpy(verifier_nonzeros).to(device))).cpu())
                        if args.dataset_name == "college_physics":
                            pred_probab = torch.nn.Softmax(dim=1)(logits)
                            y_pred = pred_probab.argmax(1).cpu()
                        else:
                            grader_actual_preds.extend([1 if prob>=0.5 else 0 for prob in logits.detach().cpu().numpy()])
                    else:
                        y_pred = []
                    if args.dataset_name == "college_physics":
                        all_preds = [0] * labels.shape[0]
                        for index, pred in zip(verifier_nonzeros, y_pred):
                            all_preds[index] = pred.item()
                        
                        for id,pred,gt in zip(report_IDs, all_preds, labels.cpu()):
                            act_gt = gt.item()
                            act_pred = pred 
                            if id in student_scores:
                                cum_preds, cum_gts = student_scores[id]
                                student_scores[id] = (cum_preds + act_pred, cum_gts + act_gt)
                            else:
                                student_scores[id] = (act_pred, act_gt)
                print("val accuracy of verifier", acc(ver_preds, ver_ground_truths))

                try:
                    ground_truths = torch.stack(ground_truths)
                    predictions = torch.stack(predictions)
                    val_verifier_loss = running_verifier_loss_val/len(val_loader.sampler)
                    if args.loss_function == "oll":
                        val_loss = criterion(predictions, ground_truths, alpha=float(args.oll_loss_alpha)).cpu().numpy()
                    else:
                        val_loss = criterion(predictions, ground_truths).cpu().numpy()
                except:
                    val_loss =10000
                if args.dataset_name == "college_physics":
                    predictions = torch.tensor([pred for pred,_ in student_scores.values()])
                    ground_truths = torch.tensor([gt for _,gt in student_scores.values()])
                else:
                    predictions = torch.tensor(grader_actual_preds)
                    ground_truths = torch.tensor(ground_truths)
                val_spearman = spearman(predictions, ground_truths, torch.ones_like(predictions))
                x = get_eval_metrics(predictions, ground_truths, torch.ones_like(predictions))
                print("val krippendorf: ",x["krippendorff_alpha"])
                print("val mse: ",x["MSE"])

            if best_val_loss_grader > val_loss:
                best_spearman = val_spearman
                best_epoch = e
                best_val_loss_grader = val_loss
                best_grader = deepcopy(grader.state_dict())
            if best_val_loss_verifier > val_verifier_loss:
                best_verifier = deepcopy(verifier.state_dict())
                best_val_loss_verifier = val_verifier_loss
           
            print("train loss: ",running_loss/len(train_loader.sampler), "val grader loss: ", val_loss, "spearman: ", val_spearman)

        if args.dataset_name == "college_physics":
            folder = "results_college"
        else:
            folder = "results_middle"
        model_save_path = f"{folder}/{args.top_k}_grader_{args.grader_model}_verifier_{args.verifier_model}_loss_{loss_function_name}_{config['lr']}_{config['batch_size']}_grader"
        os.makedirs(model_save_path)
        torch.save(best_grader, f"{model_save_path}/model.pth")
        model_save_path = f"{folder}/{args.top_k}_grader_{args.grader_model}_verifier_{args.verifier_model}_loss_{loss_function_name}_{config['lr']}_{config['batch_size']}_verifier"
        os.makedirs(model_save_path)
        torch.save(best_verifier, f"{model_save_path}/model.pth")