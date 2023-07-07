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
        self.trainer = transformers.Trainer(model=self.model)
        self.model.to('cuda')

    def train(self, train_set):
        # use the trainer to train the model on the train_set
        # the trainer will use the model and the train_set to train the model
        self.trainer.train(train_set) # train_set is a Dataset object
        pass

    def predict(self, sentences):
        for sentence in sentences['text']:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to('cuda')
            outputs = self.model(**inputs)
            yield np.argmax(outputs[0].to('cpu').detach().numpy().tolist()[0])
    
    def get_embeddings(self, sentences):
        for sentence in sentences['text']:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to('cuda')
            outputs = self.model(**inputs)
            yield torch.mean(outputs[1][-1][0], dim=0).to('cpu').detach().numpy().tolist() # last hidden state
    
    def get_probabilities(self, sentences):
        for sentence in sentences['text']:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to('cuda')
            outputs = self.model(**inputs)
            yield self.softmax(outputs[0]).to('cpu').detach().numpy().tolist()[0]
    