from transformers import AutoTokenizer, AutoModelForCausalLM

# checkpoint = "gpt2" # modelo que só completa as frases que você inputa
print("Carregando modelo...")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer = AutoTokenizer.from_pretrained("/Users/hissapinto/Library/CloudStorage/OneDrive-Pessoal/_CC/IC/HuggingFace/CodeSnippets/2_Models/gpt2")
model = AutoModelForCausalLM.from_pretrained("/Users/hissapinto/Library/CloudStorage/OneDrive-Pessoal/_CC/IC/HuggingFace/CodeSnippets/2_Models/gpt2")

print("Modelo carregado! Digite um texto e pressione Enter.")
print("Digite 'exit' para encerrar.\n")

while True:
    prompt = input(">> ")

    if prompt.strip().lower() == "exit":
        print("See you soon!")
        break

    inputs = tokenizer(prompt, return_tensors="pt")
    # print(inputs)
    # print(tokenizer.decode(inputs["input_ids"]))

    output_ids = model.generate(
        **inputs,
        max_length=60,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,      # adiciona variação nas respostas
        temperature=0.8,     # 0 = mais previsível, 1+ = mais "criativo"/aleatório
    )

    texto_gerado = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(texto_gerado)
    print()

# salva o modelo
model.save_pretrained("/Users/hissapinto/Library/CloudStorage/OneDrive-Pessoal/_CC/IC/HuggingFace/CodeSnippets/2_Models/gpt2")
tokenizer.save_pretrained("/Users/hissapinto/Library/CloudStorage/OneDrive-Pessoal/_CC/IC/HuggingFace/CodeSnippets/2_Models/gpt2")

# This will save two files to your disk:
# ls directory_on_my_computer
# config.json model.safetensors

# If you look inside the config.json file, you’ll see all the necessary attributes needed to build the model
# architecture. This file also contains some metadata, such as where the checkpoint originated and what
# 🤗 Transformers version you were using when you last saved the checkpoint.

# The pytorch_model.safetensors file is known as the state dictionary; it contains all your model’s weights.
# The two files work together: the configuration file is needed to know about the model architecture,
# while the model weights are the parameters of the model.