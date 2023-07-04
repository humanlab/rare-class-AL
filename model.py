import transformers

## define an abstract Model class that has a predict() method to return the prediction of a sentence
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        #self.model = transformers.AutoModel.from_pretrained(model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    
    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs[0].detach().numpy().tolist()[0]
    