from transformers import AutoTokenizer

# Encoding
# To get a better understanding of the two steps, we’ll explore them separately.
# Note that we will use some methods that perform parts of the tokenization pipeline
# separately to show you the intermediate results of those steps, but in practice,
# you should call the tokenizer directly on your inputs. - tokenizer("Using a Transformer network is simple")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print("\n\n" + str(tokens))

"""Saída:
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']"""

ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n\n" + str(ids))

"""Saída:
[7993, 170, 11303, 1200, 2443, 1110, 3014]"""