import os
import numpy as np
from nltk.tokenize import sent_tokenize
import torch 
import utils
#from tokenization import BertWordPieceTokenizer
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
class Verifier(torch.nn.Module):
    def __init__(self,
                 max_sentence_length=122,
                 max_question_length=42,
                 topK=3,
                 bert_or_sbert="bert"): 
        super(Verifier, self).__init__()
        cache_dir = None
        self.max_sentence_length = max_sentence_length
        self.max_question_length = max_question_length
        self.topK = topK

        if bert_or_sbert == "bert":
            self.query_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir) 
            self.context_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir) 
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                truncation_side="right", padding_side="right", cache_dir=cache_dir)
            self.model_name = "bert"
        elif bert_or_sbert == "sbert":
            self.query_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=cache_dir) #
            self.context_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=cache_dir) #
            self.model_name = "sbert"
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=cache_dir)
        else:
            raise Exception("Unknown model name for the verifier!")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    #Mean Pooling - Take attention mask into account for correct averaging. This is for SBERTS
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)
    
    
    def _preprocess(self, question, sentence, max_sentence_length=None):
        if max_sentence_length is None: # I added this because for verifier sentence length should different
            max_sentence_length = self.max_sentence_length
        tokenized_sentence = self.tokenizer(sentence, truncation=True, padding="max_length", return_tensors='pt', max_length=max_sentence_length).to(device)
        tokenized_question = self.tokenizer(question, truncation=True, padding="max_length", return_tensors='pt', max_length=self.max_question_length).to(device)
        return tokenized_question, tokenized_sentence

    def verifier_embeddings(self, questions, reports):
        questions_batch = []
        sentences_batch = []
        raw_sentences_batch = []
        max_number_of_sentences = max([len(sent_tokenize(report)) for report in reports])
        for question, report in zip(questions, reports): # basically iterates over batch
            sentences = sent_tokenize(report)
            if len(sentences) < max_number_of_sentences:
                sentences.extend([""]* (max_number_of_sentences - len(sentences))) # padding so that we have same number of sentences for each report
            questions_batch.append(question)
            sentences_batch.extend(sentences)
            raw_sentences_batch.append(sentences)
        questions_batch, sentences_batch = self.forward(questions_batch, sentences_batch)
        return questions_batch, sentences_batch

    def verify(self, questions, reports):
        questions_batch = []
        sentences_batch = []
        raw_sentences_batch = []
        max_number_of_sentences = max([len(sent_tokenize(report)) for report in reports])
        for question, report in zip(questions, reports): # basically iterates over batch
            sentences = sent_tokenize(report)
            if len(sentences) < max_number_of_sentences:
                sentences.extend([""]* (max_number_of_sentences - len(sentences))) # padding so that we have same number of sentences for each report
            questions_batch.append(question)
            sentences_batch.extend(sentences)
            raw_sentences_batch.append(sentences)
        questions_batch, sentences_batch = self.forward(questions_batch, sentences_batch)
        raw_sentences_batch = np.array(raw_sentences_batch, dtype=object)

        topK_sentences, topK_similarities, all_sims = utils.get_topK(self.topK, questions_batch, sentences_batch, raw_sentences_batch)
        if topK_similarities.shape[0]==1:
            return utils.my_sigmoid(topK_similarities.mean(dim=1)), topK_sentences, all_sims
        return utils.my_sigmoid(topK_similarities.mean(dim=1).squeeze()), topK_sentences, all_sims
    
    def forward(self, question, sentence, max_sentence_length=None):
        """Run a forward pass for each of the models and return the respective embeddings."""
        batch_size = len(question)
        num_of_sentence_per_document = len(sentence) // batch_size
        if self.model_name == "sbert":
            if max_sentence_length is None: # I added this because for verifier sentence length should different
                max_sentence_length = self.max_sentence_length

            question_dict, context_dict = self._preprocess(question, sentence, max_sentence_length)
            query_logits = self.query_model(**question_dict)
            query_logits = self._mean_pooling(query_logits, question_dict["attention_mask"])

            context_logits = self.context_model(**context_dict)
            context_logits = self._mean_pooling(context_logits, context_dict["attention_mask"])
            return query_logits, context_logits.view(batch_size, num_of_sentence_per_document, -1)
        else:
            question_dict, context_dict = self._preprocess(question, sentence, max_sentence_length)

            query_logits = self.query_model(**question_dict)
            query_logits_cls = query_logits[1]

            context_logits = self.context_model(**context_dict)
            context_logits_cls = context_logits[1]

            return query_logits_cls, context_logits_cls.view(batch_size, num_of_sentence_per_document, -1)