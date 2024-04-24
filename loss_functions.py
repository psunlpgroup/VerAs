import torch
import torch.nn.functional as F

def oll_loss(logits, labels, weights=None, alpha=3):
    num_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device)
    labels = labels.to(device)
    model_probabs = torch.nn.Softmax(dim=1)(logits)
    true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
    label_ids = len(labels)*[[k for k in range(num_classes)]]
    distances = [[float(true_labels[j][i] - label_ids[j][i]) for i in range(num_classes)] for j in range(len(labels))]
    distances_tensor = torch.tensor(distances,device=device, requires_grad=True)
    if weights is not None:
        err = (-torch.log(1-model_probabs)*abs(distances_tensor)**alpha) * weights
    else:
        err = (-torch.log(1-model_probabs)*abs(distances_tensor)**alpha)
    loss = torch.sum(err,axis=1).mean()
    return loss

def cross_entropy_loss(logits, labels, weights=None):
    if weights is None:
        loss = torch.nn.functional.cross_entropy(logits, labels)
    else:
        num_classes = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logits = logits.to(device)
        labels = labels.to(device)
        model_probabs = torch.nn.Softmax(dim=1)(logits)
        log_probs = -torch.log(model_probabs)
        one_hot_labels= torch.nn.functional.one_hot(labels, num_classes=num_classes)
        loss = torch.mul(log_probs, one_hot_labels) * weights
        loss = torch.sum(loss, 1).mean()
    return loss

def mse(logits, labels):
    return torch.nn.functional.mse_loss(logits, labels)
