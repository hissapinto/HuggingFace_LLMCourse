# exemplo de uso de padding gerado pelo claude.
# Bom exemplo prático: gerar continuações para várias frases de uma vez, em vez de uma por uma.
# Isso é comum quando você quer processar um monte de prompts rapidamente (ex: gerar respostas
# para uma lista de perguntas de um dataset), ao invés de rodar o modelo dezenas de vezes em loop.
# paralelismo acontece

from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 não tem pad_token nativo

model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Várias sequências de tamanhos DIFERENTES, processadas de uma vez só (batch)
prompts = [
    "Hello! It's nice to",
    "The weather today is",
    "In the future, artificial intelligence will probably",
]

# padding=True é OBRIGATÓRIO aqui: sem ele, o tokenizer não consegue
# criar um tensor retangular com frases de tamanhos diferentes
inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

print("\n\ninput_ids (repare nos zeros de padding preenchendo as frases mais curtas):")
print(inputs["input_ids"])
print()
print("attention_mask (0 = ignorar, é só padding):")
print(inputs["attention_mask"])
print()

output_ids = model.generate(
    **inputs,
    max_new_tokens=15,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.2,
)

print("\n--- Resultados ---")
for i, ids in enumerate(output_ids):
    texto = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"[{i}] {texto}")
