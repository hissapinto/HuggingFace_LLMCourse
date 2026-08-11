import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# ids precisa ser [] pois o metodo espera mais de uma sequence por default,
# então, mesmo se for só uma, ela precisa ser envelopada em uma dimensão extra
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

"""Saída:
    Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
    Logits: [[-2.7276,  2.8789]]
"""

# Batching is the act of sending multiple sentences through the model, all at once.
# If you only have one sentence, you can just build a batch with a single sequence:
batched_ids = [ids, ids]

# Try it out! Convert this batched_ids list into a tensor and pass it through your model.
# Check that you obtain the same logits as before (but twice)!

input_ids_batch = torch.tensor(batched_ids)
print("Input IDs:", input_ids_batch)

output_batches = model(input_ids_batch)
print("Logits:", output_batches.logits)

# Pra transformar os ids em tensors, os vetores devem ter o mesmo tamanho (multiplicação
# de matriz), sendo assim, precisamos de padding para completar os vetores desiguais

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print("------------------------------------------------------------------------------")
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print("------------------------------------------------------------------------------")
print(model(torch.tensor(batched_ids)).logits)

"""Saída:
    tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
    tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)

    tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
"""

# There’s something wrong with the logits in our batched predictions: the second row
# should be the same as the logits for the second sentence, but we’ve got completely
# different values!

# This is because the key feature of Transformer models is attention layers that
# contextualize each token. These will take into account the padding tokens since
# they attend to all of the tokens of a sequence. To get the same result when passing
# individual sentences of different lengths through the model or when passing a batch
# with the same sentences and padding applied, we need to tell those attention layers
# to ignore the padding tokens. This is done by using an attention mask.

# Attention masks
# Attention masks are tensors with the exact same shape as the input IDs tensor,
# filled with 0s and 1s. 0 = they should be ignored by the attention layers of the model.

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print("------------------------------------------------------------------------------")
print(outputs.logits)

"""Saída:
    tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
"""

# Longer sentences
# With Transformer models, there is a limit to the lengths of the sequences we can
# pass the models. Most models handle sequences of up to 512 or 1024 tokens,
# and will crash when asked to process longer sequences.
# There are two solutions to this problem:

#    Use a model with a longer supported sequence length.
#    Truncate your sequences.

# we recommend you truncate your sequences by specifying the max_sequence_length parameter:

max_sequence_length = 1024
sequence = sequence[:max_sequence_length]