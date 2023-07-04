import transformers
import torch
import numpy as np
## define a Model class that has a predict(), get_probabilities() and get_embedding() method to return the prediction, probabilities of each class, and the embedding respectively of a sentence
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.softmax = torch.nn.Softmax(dim=1)

    def train(self, train_set):
        pass

    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return np.argmax(outputs[0].detach().numpy().tolist()[0])
    
    def get_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return torch.mean(outputs[1][-1][0], dim=0).detach().numpy().tolist() # last hidden state
    
    def get_probabilities(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return self.softmax(outputs[0]).detach().numpy().tolist()[0]
    