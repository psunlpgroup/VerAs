import math
import numpy as np
import random
import torch


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def my_sigmoid(tensor, k=10):
    return 1/(1+torch.exp(-k*(tensor-0.5)))
    

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def bert_attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

# THIS ONE WORKS WITH BATCH
def get_topK(topK, query_logits, sentence_logits, raw_sentences):
    similarities = torch.cosine_similarity(query_logits.unsqueeze(1), sentence_logits, dim=2)
    topK_sentence_similarity, topK_sentence_indicies = torch.topk(similarities, k=min(topK, similarities.shape[1]), dim=1)
    return ["".join(np.take(sentences, indices)) for sentences,indices in zip(raw_sentences, topK_sentence_indicies.cpu().numpy())], topK_sentence_similarity, similarities.cpu().detach().numpy()
