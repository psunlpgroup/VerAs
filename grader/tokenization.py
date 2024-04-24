import math
from transformers import ElectraTokenizer, LongformerTokenizer, BertTokenizer, AutoTokenizer
import torch

class Tokenizer(object):
    def __init__(self, 
        model_type:str,
        ) -> None:
        cache_dir = None
        if model_type == "electra":
            self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator", cache_dir=cache_dir)
        elif model_type == "longformer":
            self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=cache_dir)
        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
        elif model_type == "longt5":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=cache_dir)
        else:
            raise Exception("unknown model type from tokenizer!")

    def __call__(self, questions, reports, topKSentences):
        return self.tokenizer(questions, truncation=True, padding="max_length", return_tensors='pt'), self.tokenizer(reports, truncation=True, padding="max_length", return_tensors="pt"), self.tokenizer(topKSentences, truncation=True, padding="max_length", return_tensors='pt')
    
    def get_vocab_len(self):
        return len(self.tokenizer)
