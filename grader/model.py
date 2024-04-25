import torch
from transformers import ElectraModel, BertModel, LongT5EncoderModel

class Grader(torch.nn.Module):
    def __init__(self, model_type="electra", dataset_name="college_physics"):
        super().__init__()
        cache_dir = None
        if model_type == "electra":
            self.query_model = ElectraModel.from_pretrained('google/electra-small-discriminator', cache_dir=cache_dir) 
            self.document_model = ElectraModel.from_pretrained('google/electra-small-discriminator', cache_dir=cache_dir) 
            self.model_name = "electra"
        elif model_type == "bert":
            self.query_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir) 
            self.document_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir) 
            self.model_name = "bert"
        elif model_type == "longt5":
            self.query_model = LongT5EncoderModel.from_pretrained("google/long-t5-tglobal-base", cache_dir=cache_dir)
            self.document_model = LongT5EncoderModel.from_pretrained("google/long-t5-tglobal-base", cache_dir=cache_dir)
            self.model_name = "longt5"
        else:
            raise Exception("Unknown model name for the grader!")
        if dataset_name == "college_physics":
            self.dense = torch.nn.Linear(self.query_model.config.hidden_size*3,6) # hidden state size *3 because 1 for query 1 for document, 1 for topSentence, output 6 because we have 6 classes
        else:
            self.dense = torch.nn.Linear(self.query_model.config.hidden_size*3,1)
        
        self.dataset_name = dataset_name

    def forward(self, query_dict, reports_dict, topSentence_dict):
        batch_size = reports_dict["input_ids"].shape[0]
        query_embeddings = self.query_model(**query_dict)
        for key in reports_dict:
            reports_dict[key] = reports_dict[key].view(-1,reports_dict[key].shape[-1]) # normally its shape is btchsize x number_of_windows x model_size
        report_embeddings = self.document_model(**reports_dict)
        report_embeddings_last_hidden_avg = report_embeddings[0].view(batch_size, -1, report_embeddings[0].shape[-2],report_embeddings[0].shape[-1]).mean(dim=1)
        topSentence_embeddings = self.document_model(**topSentence_dict)
        
        combined = torch.cat((query_embeddings[0][:,0,:], report_embeddings_last_hidden_avg[:,0,:], topSentence_embeddings[0][:,0,:]), 1)
        if self.dataset_name == "college_physics":
            return self.dense(combined)
        else:
            return torch.sigmoid(self.dense(combined)).view(batch_size)
