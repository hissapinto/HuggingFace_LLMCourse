# O processo:
# frase -> tokens -> ids -> trucation (if needed) ->
# padding (if needed) -> attention mask -> embeddings (ids viram vetores, sem contexto)
# -> layers/atenção (vetores absorvem contexto entre si -> hidden states) ->
# head de classificação (projeta hidden states para o espaço de classes) ->
# logits (saída bruta da head, ainda não são probabilidades)

# hidden states: São os vetores que representam cada token depois de passar pelas camadas
# de atenção — ou seja, já carregados de contexto, mas ainda sem nenhuma interpretação de
# tarefa aplicada em cima.

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# SEM head - só o corpo, para em "hidden states"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print("------------------------------------------------------------------------------")
print(outputs.last_hidden_state.shape)   # [2, 16, 768]


# COM head - corpo + head de classificação, chega em "logits"
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model2(**inputs)
print(outputs.logits.shape)   # [2, 2]
print("------------------------------------------------------------------------------")

# Putting it all together
# In the last few sections, we’ve been trying our best to do most of the work by hand.
# We’ve explored how tokenizers work and looked at tokenization, conversion to input IDs,
# padding, truncation, and attention masks.

# However, as we saw in section 2, the 🤗 Transformers API can handle all of this for us
# with a high-level function that we’ll dive into here. When you call your tokenizer
# directly on the sentence, you get back inputs that are ready to pass through your model:

tokenizer2 = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer2(sequence)
print(model_inputs)
print("------------------------------------------------------------------------------")

"""Saída:
    {'input_ids': [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

# Here, the model_inputs variable contains everything that’s necessary for a model to operate
# well. For DistilBERT, that includes the input IDs as well as the attention mask. Other models
# that accept additional inputs will also have those output by the tokenizer object.

# It also handles multiple sequences at a time, with no change in the API:

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs2 = tokenizer(sequences)
print(model_inputs2)
print("------------------------------------------------------------------------------")

"""Saída:
    {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], [101, 2061, 2031, 1045, 999, 102]],
    'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}
"""

# The tokenizer object can handle the conversion to specific framework tensors,
# which can then be directly sent to the model. For example, in the following code sample
# we are prompting the tokenizer to return tensors from the different frameworks — "pt"
# returns PyTorch tensors and "np" returns NumPy arrays:

sequences_conv = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs_pt = tokenizer(sequences_conv, padding=True, return_tensors="pt")
# Returns NumPy arrays
model_inputs_np = tokenizer(sequences_conv, padding=True, return_tensors="np")

# Special tokens
# If we take a look at the input IDs returned by the tokenizer, we will see they are a
# tiny bit different from what we had earlier:

sequence_st = "I've been waiting for a HuggingFace course my whole life."

model_inputs_st = tokenizer(sequence_st)
print(model_inputs_st["input_ids"])

tokens_st = tokenizer.tokenize(sequence_st)
ids_st = tokenizer.convert_tokens_to_ids(tokens_st)
print(ids_st)
print("------------------------------------------------------------------------------")

"""Saída:
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
         [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
"""

print(tokenizer.decode(model_inputs_st["input_ids"]))
print(tokenizer.decode(ids_st))
print("------------------------------------------------------------------------------")

"""Saída""
    "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
          "i've been waiting for a huggingface course my whole life."
"""

# The tokenizer added the special word [CLS] at the beginning and the special word [SEP] at the
# end. This is because the model was pretrained with those, so to get the same results for
# inference we need to add them as well. Note that some models don’t add special words,
# or add different ones; models may also add these special words only at the beginning,
# or only at the end. In any case, the tokenizer knows which ones are expected and will deal
# with this for you.

# Wrapping up: From tokenizer to model
# Now that we’ve seen all the individual steps the tokenizer object uses when applied on texts,
# let’s see one final time how it can handle multiple sequences (padding!), very long sequences
# (truncation!), and multiple types of tensors with its main API:

checkpoint_final = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_final = AutoTokenizer.from_pretrained(checkpoint_final)
model_final = AutoModelForSequenceClassification.from_pretrained(checkpoint_final)
sequences_final = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens_final = tokenizer(sequences_final, padding=True, truncation=True, return_tensors="pt")
output = model_final(**tokens_final)
print(output)

"""Saída:
    SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
    [-3.6183,  3.9137]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""