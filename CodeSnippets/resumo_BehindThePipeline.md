# Guia: Behind the Pipeline (Hugging Face Course)

## Setup do ambiente

```bash
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate
pip install -r requirements.txt
python seu_script.py
```

`requirements.txt`:
```
transformers>=4.40.0
torch>=2.0.0
```

---

## O fluxo completo do pipeline

```
Model input --> [ Embeddings --> Layers ] --> Hidden states --> Head --> Model output
                 \____________________/
                    Transformer network
\_______________________________________________________/
                    Full model
```

- **Embeddings**: tabela de consulta (id → vetor). Cada id sempre gera o **mesmo vetor**, não importa o contexto. É treinável, mas inicialmente aleatória.
- **Layers**: aplicam atenção — cada token "olha" para os vizinhos e absorve contexto. É aqui que o vetor deixa de ser estático e vira contextual.
- **Hidden states**: saída das layers. Vetores ricos em contexto, mas ainda sem interpretação de tarefa.
- **Head**: camada(s) linear(es) extra que projeta os hidden states para o formato da tarefa (ex: 2 classes para sentimento).

---

## Passo 1 — Tokenização

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
```

Saída: um dicionário com dois tensores.

```
'input_ids':      tensor([[101, 1045, ...,   102],
                           [101, 1045, ...,     0]])   <- ids dos tokens (padding = 0)

'attention_mask': tensor([[1, 1, ..., 1],
                           [1, 1, ..., 0, 0, 0]])       <- 1 = real, 0 = ignorar (padding)
```

- `input_ids`: tensor 2D (frases × tokens), cada número é o id de um token no vocabulário.
- `attention_mask`: diz ao modelo quais posições são texto real (1) e quais são só preenchimento (0), para a atenção ignorar o padding.

---

## Passo 2 — Corpo do modelo (sem head): `AutoModel`

```python
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
outputs.last_hidden_state.shape   # torch.Size([2, 16, 768])
```

```
[2, 16, 768]
 |   |    |
 |   |    +-- 768 números por token (embedding contextual)
 |   +------- 16 tokens por frase (após padding)
 +----------- 2 frases
```

Isso é o resultado de **Embeddings + Layers**, parando em "Hidden states" — ainda sem nenhuma tarefa aplicada.

> Ao carregar `AutoModel` a partir de um checkpoint de classificação, o `transformers` avisa que os pesos `classifier` e `pre_classifier` são "UNEXPECTED" — é esperado, pois `AutoModel` descarta a head.

---

## Passo 3 — Corpo + head de classificação: `AutoModelForSequenceClassification`

```python
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model2(**inputs)
outputs.logits.shape   # torch.Size([2, 2])
```

```
Hidden states [2, 16, 768] --> Head (linear) --> Logits [2, 2]
                                                   2 frases, 2 classes
```

`logits` são valores **brutos**, não normalizados — ainda não são probabilidades:

```python
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]])
```

---

## Passo 4 — Softmax: transformar logits em probabilidades

```python
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

```
tensor([[4.0195e-02, 9.5980e-01],   # frase 1: ~96% positiva
        [9.9946e-01, 5.4418e-04]])  # frase 2: ~99.9% negativa
```

## Passo 5 — Mapear posição → rótulo

```python
model2.config.id2label
# {0: 'NEGATIVE', 1: 'POSITIVE'}
```

---

## Resumo mental rápido

| Etapa | Classe/função | Shape de saída | O que representa |
|---|---|---|---|
| Tokenizar | `AutoTokenizer` | `[2, 16]` (ids) | Texto → números |
| Corpo do modelo | `AutoModel` | `[2, 16, 768]` | Vetores contextuais por token |
| Corpo + head | `AutoModelForSequenceClassification` | `[2, 2]` | Score bruto por classe (logits) |
| Softmax | `torch.nn.functional.softmax` | `[2, 2]` | Probabilidades (somam 1) |
| Rótulos | `model.config.id2label` | dict | Nome de cada classe |
