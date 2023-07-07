## Description: This file contains functions to extract embeddings from a huggingface transformer model
## use a huggingface transformer to extract embeddings to a file
## this is a modified version of the script from
## https://huggingface.co/transformers/quickstart.html#using-the-pipeline

import transformers

#get_embedding() function to return the embedding of a sentence 

def get_embedding(model_name, sentence):
    # load the model
    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # encode the line
    inputs = tokenizer(sentence, return_tensors="pt")
    # get the embedding
    outputs = model(**inputs)
    # return the embedding
    return outputs[0].detach().numpy().tolist()[0]


def extract_embeddings_to_file(model_name, input_file, output_file):

    # load the data
    with open(input_file, "r") as f:
        lines = f.readlines()

    # extract embeddings
    with open(output_file, "w") as f:
        for line in lines:
            # encode the line and write the embedding to file
            f.write(str(get_embedding(model_name, line)) + "\n")


