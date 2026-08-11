# Optimized Inference Deployment

Resumo dos mecanismos usados para rodar LLMs de forma eficiente — do treino até dispositivos com pouco hardware.

---

## 1. Contexto: por que isso importa

Rodar um LLM não é só "fazer a conta" — é lidar com **gargalos de memória e escala**. Três problemas diferentes motivam três soluções diferentes:

| Problema | Onde aparece | Solução |
|---|---|---|
| O cálculo de atenção é lento por causa de transferências de memória | Dentro de uma única operação de atenção | **Flash Attention** |
| Memória do cache (KV cache) é desperdiçada/fragmentada entre requisições | Servindo múltiplos usuários ao mesmo tempo | **PagedAttention (vLLM)** |
| O modelo nem cabe no hardware disponível | Hardware limitado (CPU, edge, Raspberry Pi) | **Quantização (llama.cpp)** |

Essas três técnicas **não competem entre si** — atacam camadas diferentes do mesmo problema geral e podem ser usadas juntas.

---

## 2. Flash Attention

### As letras da fórmula de atenção (Q, K, V, S, P, O)

Antes do mecanismo de Flash Attention em si, vale fixar o que cada letra representa — são vetores calculados **para cada token**, a partir do hidden state dele, usando três matrizes de peso diferentes e treináveis (`W_Q`, `W_K`, `W_V`):

| Letra | Nome | O que é | Papel |
|---|---|---|---|
| **Q** | Query | Vetor: "o que eu, token, preciso saber sobre o contexto" | Usado para **comparar** com as Keys de todos os tokens |
| **K** | Key | Vetor: "a etiqueta que descreve o que eu, token, ofereço" | Usado para **ser comparado** com a Query — só decide "o quanto" um token é relevante, nunca entra no resultado final diretamente |
| **V** | Value | Vetor: "o conteúdo real que eu, token, carrego" | É o que efetivamente compõe o resultado final, ponderado pela relevância calculada |
| **S** | Scores | `S = QKᵀ` — scores brutos de similaridade entre cada Query e cada Key | Números crus (podem ser negativos), ainda não interpretáveis como "pesos" |
| **P** | Probabilities | `P = softmax(S)` — os scores normalizados | Números entre 0 e 1 que somam 1: a fração de atenção que cada token merece |
| **O** | Output | `O = PV` — combinação ponderada de todos os Values, usando os pesos de P | O resultado final da camada de atenção para aquele token |

**Fluxo completo:** `Q` e `K` se comparam (`S`) → `S` vira pesos via softmax (`P`) → `P` pondera os `V` → resultado (`O`).

**Analogia da biblioteca:** Query é sua pergunta; Key é a etiqueta de cada livro (só serve para decidir relevância); Value é o conteúdo real do livro (o que você efetivamente absorve, na proporção da relevância calculada). Key nunca "é" a resposta — só ajuda a decidir quanto peso dar a cada Value.

### O problema: gargalo de memória, não de cálculo

GPUs têm dois tipos de memória:
- **HBM** (High Bandwidth Memory): grande, mas relativamente lenta
- **SRAM**: cache do chip, minúsculo, mas muito rápido

A atenção tem complexidade **quadrática** (a matriz de atenção cresce com o quadrado do tamanho da sequência). O gargalo real não é a conta em si — é o **transporte de dados** entre HBM e SRAM a cada etapa.

### Standard Attention (ineficiente)

```
Load Q,K  -> calcula S = QKᵀ          -> Write S (volta pra memória)
Load S    -> calcula P = softmax(S)   -> Write P (volta pra memória)
Load P,V  -> calcula O = PV           -> Write O (volta pra memória)
```

6 transferências HBM↔SRAM para **uma única** operação de atenção. A GPU fica ociosa esperando dados.

### Flash Attention (otimizado)

- Divide Q, K, V em **blocos** pequenos o suficiente para caber na SRAM
- Processa cada bloco **inteiramente dentro da SRAM** (sem sair no meio do caminho)
- Usa "softmax online": mantém acumuladores `m` (máximo) e `l` (soma), atualizados a cada bloco, para calcular o softmax de forma incremental sem precisar da sequência inteira de uma vez
- Só escreve o resultado (`O`, `l`, `m`) na HBM ao final de cada bloco

**Analogia:** em vez de fazer o bolo inteiro numa bancada pequena (não cabe), você faz fatia por fatia — pega os ingredientes da geladeira (HBM), monta a fatia inteira na bancada (SRAM) sem voltar à geladeira no meio, guarda a fatia pronta, repete.

### Treino vs. Inferência

| | Treino | Inferência |
|---|---|---|
| Usa Flash Attention? | Sim | Sim |
| Impacto | **Grande** — forward + backward, lotes grandes, repetido por muitas iterações | Moderado, mas real — especialmente com contexto longo ou muitos usuários simultâneos |

---

## 3. PagedAttention (vLLM)

### O que é o KV Cache (contexto necessário)

Na geração autoregressiva (token a token), recalcular a atenção do zero a cada novo token seria um desperdício. Por isso, os vetores `K` e `V` já calculados de tokens anteriores são **guardados em cache** — o **KV cache** — e reaproveitados a cada novo passo.

> Nota: apesar do nome, K/V não são um "dicionário" de busca exata. Key serve para **comparação** (via softmax), Value é o **conteúdo** que compõe o resultado final como uma média ponderada.

### O problema do KV cache tradicional

- Cada requisição reserva um bloco de memória **contínuo**, geralmente dimensionado para o "pior caso" (tamanho máximo possível)
- Isso causa **desperdício** (memória reservada e não usada) e **fragmentação** (memória livre espalhada em pedaços pequenos demais para novas requisições)

### A solução: paginação, como em sistemas operacionais

PagedAttention aplica a mesma ideia que SOs usam para gerenciar RAM há décadas: em vez de blocos contínuos, o KV cache é dividido em **páginas/blocos pequenos e fixos**, que podem ficar espalhados fisicamente na memória e são referenciados por um mapa lógico.

Benefícios:
- Sem desperdício de memória reservada à toa
- Sem fragmentação
- Requisições com prefixos iguais (ex: mesmo prompt de sistema) podem **compartilhar blocos**, em vez de duplicar

### Throughput vs. Latência

| | Latência | Throughput |
|---|---|---|
| Pergunta | "Quanto tempo até eu receber minha resposta?" | "Quantas respostas o sistema produz por segundo, no total?" |
| Escopo | Uma requisição | O sistema inteiro, várias requisições |

PagedAttention ajuda principalmente o **throughput**: como desperdiça menos memória, mais requisições cabem simultaneamente na mesma GPU — daí ganhos relatados de até 24x em throughput.

---

## 4. llama.cpp e Quantização

### Cenário diferente dos anteriores

Flash Attention e PagedAttention assumem uma GPU de servidor já disponível. **llama.cpp** resolve um problema anterior: **o modelo cabe no hardware que você tem?** (ex: CPU, Raspberry Pi, edge devices).

### Quantização: a técnica central

Reduz a precisão numérica dos pesos do modelo:

```
FP32 (32 bits) -> FP16 (16 bits) -> INT8 (8 bits) -> 4-bit -> 2-bit
```

Menos bits por peso = modelo ocupa muito menos memória, ao custo de perder um pouco de precisão numérica (trade-off: mais compressão = mais erro acumulado, especialmente abaixo de 4-bit).

### Características principais

- **Múltiplos níveis de quantização**: 8-bit, 4-bit, 3-bit, 2-bit
- **Formato GGML/GGUF**: formato de tensor customizado, otimizado para inferência quantizada
- **Precisão mista**: partes diferentes do modelo podem usar níveis de quantização diferentes
- **Otimizações específicas de hardware**: código otimizado para AVX2, AVX-512 (CPUs x86), NEON (ARM, como no Raspberry Pi)

---

## 5. Comparação geral

| | Flash Attention | PagedAttention (vLLM) | llama.cpp |
|---|---|---|---|
| **Cenário alvo** | GPU, servidor | GPU, servidor, múltiplos usuários | CPU/hardware limitado, uso local |
| **O que otimiza** | Cálculo da atenção em si | Gerenciamento de memória do KV cache | Tamanho do modelo |
| **Técnica central** | Reduzir transferências HBM↔SRAM | Paginar o KV cache em blocos | Quantização dos pesos |
| **Métrica que mais melhora** | Velocidade/memória por operação de atenção | Throughput (requisições simultâneas) | Viabilidade de rodar em hardware fraco |

Um sistema de produção robusto pode combinar os três: modelo quantizado (cabe na memória) + Flash Attention (cálculo eficiente) + PagedAttention (cache bem gerenciado entre múltiplas requisições).

**Exemplo prático:** rodar Gemma 3 num Raspberry Pi via llama.cpp — cenário de único usuário, hardware limitado, sem GPU relevante. Aqui, quantização é essencial; Flash Attention e PagedAttention seriam pouco relevantes (brilham em GPUs de servidor com múltiplas requisições simultâneas, contexto bem diferente de um Pi rodando sozinho).

---

## 6. Deployment e Integração (TGI vs. vLLM vs. llama.cpp)

Além das técnicas internas, os frameworks que as implementam também diferem em **para quem/onde são pensados**:

| | TGI | vLLM | llama.cpp |
|---|---|---|---|
| **Foco** | Deployment enterprise, pronto para produção | Performance bruta e flexibilidade para devs | Simplicidade e portabilidade |
| **Recursos** | Kubernetes nativo, monitoramento (Prometheus/Grafana), auto-scaling, logging enterprise, content filtering, rate limiting | Núcleo em Python, substitui a API da OpenAI facilmente, integra bem com Ray para clusters | Core em C/C++, dependências mínimas, roda até em laptops/mobile |
| **Melhor para** | Empresas que precisam de infraestrutura de produção completa "out of the box" | Times que querem performance máxima e customização | Ambientes onde instalar frameworks Python é inviável (edge, hardware limitado) |
| **API** | — | Compatível com OpenAI | Compatível com OpenAI, footprint bem menor |

Isso conecta diretamente com as técnicas da seção 5: TGI e vLLM (GPU, servidor) tendem a se beneficiar mais de Flash Attention e PagedAttention; llama.cpp (hardware limitado) é onde a quantização é a peça central — exatamente o caso do Gemma 3 no Raspberry Pi.
