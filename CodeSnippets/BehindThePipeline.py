# criar e rodar o env
# python -m venv env
# source env/bin/activate
# pip install -r requirements.txt
# python seu_script.py


from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch

# impressão dos tensor com os ids de cada palavra
# e a máscara de atenção (1 = palavra real, 0 = ignorar, só o padding)
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print("\n\n" + str(inputs) + "\n\n")

"""saída:
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
 } """

# imprime o resultado da fase embedding + layers, que já são os vetores com contexto
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print("\n\n" + str(outputs.last_hidden_state.shape) + "\n\n")

"""  saída:
 torch.Size([2, 16, 768]) = 2 frases, 16 tokens, 768 números por token. """

# For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative).
# So, we won’t actually use the AutoModel class, but AutoModelForSequenceClassification:
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model2(**inputs)
# Now if we look at the shape of our outputs, the dimensionality will be much lower: the model head takes as input the high-dimensional vectors we saw before, and outputs vectors containing two values (one per label):
print("\n\n" + str(outputs.logits.shape) + "\n")
""" saída: torch.Size([2, 2]) """

# Our model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one. Those are not probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model. 
print("\n" + str(outputs.logits) + "\n\n")
""" saída: 
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>) """

# To be converted to probabilities, they need to go through a SoftMax layer
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print("\n\n" + str(predictions) + "\n\n")
""" saída: 
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)"""

# To get the labels corresponding to each position, we can inspect the id2label attribute of the model config (more on this in the next section):
print("\n\n" + str(model2.config.id2label) + "\n\n")
"""Saída:
{0: 'NEGATIVE', 1: 'POSITIVE'}"""