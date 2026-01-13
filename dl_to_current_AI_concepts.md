plaintext

# 🚀 MODERN AI & DEEP LEARNING - From Transformers to Today's Market

> **A comprehensive guide covering the evolution from Transformers to current state-of-the-art AI technologies, including LLMs, Generative AI, RAG, Agents, MLOps, and production-ready systems used in the industry today.**

---

## Table of Contents

1. [Evolution Recap: From RNNs to Transformers](#evolution-recap-from-rnns-to-transformers)
2. [Large Language Models (LLMs)](#large-language-models-llms)
3. [BERT and Encoder-Only Models](#bert-and-encoder-only-models)
4. [GPT and Decoder-Only Models](#gpt-and-decoder-only-models)
5. [Modern LLM Architectures](#modern-llm-architectures)
6. [Fine-Tuning Techniques](#fine-tuning-techniques)
7. [Prompt Engineering](#prompt-engineering)
8. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
9. [Vector Databases and Embeddings](#vector-databases-and-embeddings)
10. [AI Agents and Autonomous Systems](#ai-agents-and-autonomous-systems)
11. [Generative AI](#generative-ai)
12. [Vision Transformers (ViT)](#vision-transformers-vit)
13. [Multimodal Models](#multimodal-models)
14. [MLOps and Production Systems](#mlops-and-production-systems)
15. [Model Optimization and Deployment](#model-optimization-and-deployment)
16. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
17. [Safety, Alignment, and Ethics](#safety-alignment-and-ethics)
18. [Current Industry Tools and Frameworks](#current-industry-tools-and-frameworks)
19. [Future Trends](#future-trends)
20. [Quick Reference and Resources](#quick-reference-and-resources)

---

## Evolution Recap: From RNNs to Transformers

### The Journey So Far

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTION OF SEQUENCE MODELS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RNN (1986)                                                                 │
│    │ Problem: Vanishing gradients, can't capture long dependencies          │
│    ↓                                                                        │
│  LSTM (1997)                                                                │
│    │ Solution: Gates to control information flow                            │
│    │ Problem: Still sequential, slow training                               │
│    ↓                                                                        │
│  GRU (2014)                                                                 │
│    │ Simplified LSTM, faster but same sequential limitation                 │
│    ↓                                                                        │
│  Attention Mechanism (2014-2015)                                            │
│    │ Allow model to focus on relevant parts                                 │
│    │ Problem: Still uses RNNs underneath                                    │
│    ↓                                                                        │
│  Transformer (2017) - "Attention Is All You Need"                          │
│    │ Removed RNNs entirely, parallel processing                             │
│    │ Self-attention mechanism                                               │
│    ↓                                                                        │
│  BERT (2018) - Encoder-only, bidirectional                                  │
│  GPT (2018) - Decoder-only, autoregressive                                  │
│    │                                                                        │
│    ↓                                                                        │
│  GPT-2, GPT-3, GPT-4 (2019-2023)                                           │
│  LLaMA, Claude, PaLM, Gemini (2023-2024)                                   │
│    │                                                                        │
│    ↓                                                                        │
│  Modern Era: Multimodal, Agents, RAG, Fine-tuning (2024-Present)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Transformers Changed Everything

| Aspect | Before Transformers | After Transformers |
|--------|--------------------|--------------------|
| **Training** | Sequential (slow) | Parallel (fast) |
| **Context** | Limited by memory | Full attention to all tokens |
| **Transfer Learning** | Limited | Pre-train once, fine-tune everywhere |
| **Scalability** | Hard to scale | Scale with compute and data |
| **Multimodal** | Separate architectures | Unified architecture |

---

## Large Language Models (LLMs)

### What are LLMs?

Large Language Models are neural networks trained on massive text corpora that can understand, generate, and manipulate human language. They're typically based on the Transformer architecture and contain billions of parameters.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM ARCHITECTURE TYPES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENCODER-ONLY (BERT-style)                                                  │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Input: "The [MASK] sat on the mat"                         │           │
│  │                    ↓                                        │           │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │           │
│  │  │ The │ │[MSK]│ │ sat │ │ on  │ │ the │ │ mat │          │           │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘          │           │
│  │     └───────┴───────┴───────┴───────┴───────┘               │           │
│  │                    ↓ Bidirectional Attention                │           │
│  │              [Contextualized Embeddings]                    │           │
│  │                    ↓                                        │           │
│  │  Output: "cat" (fill in the mask)                          │           │
│  │                                                             │           │
│  │  Use: Classification, NER, Question Answering               │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  DECODER-ONLY (GPT-style)                                                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Input: "The cat sat on"                                    │           │
│  │                    ↓                                        │           │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                           │           │
│  │  │ The │→│ cat │→│ sat │→│ on  │→ [Predict next]           │           │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                           │           │
│  │                    ↓ Causal (Left-to-Right) Attention       │           │
│  │                                                             │           │
│  │  Output: "the" → "mat" → "." (autoregressive generation)   │           │
│  │                                                             │           │
│  │  Use: Text Generation, Chatbots, Code Generation            │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  ENCODER-DECODER (T5, BART-style)                                          │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Input: "Translate: The cat sat on the mat"                 │           │
│  │                    ↓                                        │           │
│  │           [ENCODER] → [Context Vector] → [DECODER]          │           │
│  │                    ↓                                        │           │
│  │  Output: "Le chat s'est assis sur le tapis"                │           │
│  │                                                             │           │
│  │  Use: Translation, Summarization, Question Answering        │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scaling Laws

LLMs follow predictable scaling laws - performance improves with:
1. **More Parameters** (model size)
2. **More Data** (training corpus)
3. **More Compute** (training FLOPs)

```
Performance ∝ (Parameters)^α × (Data)^β × (Compute)^γ

Typical values: α ≈ 0.076, β ≈ 0.095, γ ≈ 0.050
```

### Major LLM Families

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAJOR LLM FAMILIES (2024)                            │
├─────────────────┬───────────────┬───────────────────────────────────────────┤
│ Family          │ Organization  │ Key Features                              │
├─────────────────┼───────────────┼───────────────────────────────────────────┤
│ GPT-4/4o        │ OpenAI        │ Multimodal, largest capabilities          │
│ Claude 3/3.5    │ Anthropic     │ Safety-focused, long context (200K)       │
│ Gemini          │ Google        │ Multimodal native, efficient              │
│ LLaMA 2/3       │ Meta          │ Open weights, research-friendly           │
│ Mistral/Mixtral │ Mistral AI    │ Efficient, MoE architecture               │
│ Command R       │ Cohere        │ Enterprise, RAG-optimized                 │
│ Qwen            │ Alibaba       │ Multilingual, code-strong                 │
│ DeepSeek        │ DeepSeek      │ Open, competitive performance             │
└─────────────────┴───────────────┴───────────────────────────────────────────┘
```

---

## BERT and Encoder-Only Models

### BERT (Bidirectional Encoder Representations from Transformers)

**Released:** October 2018 by Google

**Key Innovation:** Bidirectional pre-training using Masked Language Modeling (MLM)

### How BERT Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BERT ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRE-TRAINING TASKS:                                                        │
│                                                                             │
│  1. Masked Language Modeling (MLM)                                          │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Input:  "The [MASK] jumped over the [MASK] dog"                 │    │
│     │ Target: "The  cat   jumped over the  lazy  dog"                 │    │
│     │                                                                 │    │
│     │ • Randomly mask 15% of tokens                                   │    │
│     │ • 80% replaced with [MASK]                                      │    │
│     │ • 10% replaced with random token                                │    │
│     │ • 10% unchanged                                                 │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Next Sentence Prediction (NSP)                                          │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Input:  [CLS] Sentence A [SEP] Sentence B [SEP]                 │    │
│     │ Output: IsNext / NotNext                                        │    │
│     │                                                                 │    │
│     │ • 50% actual next sentences                                     │    │
│     │ • 50% random sentences                                          │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  INPUT REPRESENTATION:                                                      │
│                                                                             │
│     Token Embeddings:     [CLS] The  cat  sat  [SEP] It  was  ... [SEP]   │
│           +                                                                 │
│     Segment Embeddings:    E_A  E_A  E_A  E_A  E_A   E_B E_B  ... E_B     │
│           +                                                                 │
│     Position Embeddings:   P_0  P_1  P_2  P_3  P_4   P_5 P_6  ... P_n     │
│           =                                                                 │
│     Final Input            ─────────────────────────────────────────       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### BERT Variants

| Model | Parameters | Key Difference |
|-------|------------|----------------|
| **BERT-Base** | 110M | 12 layers, 768 hidden, 12 heads |
| **BERT-Large** | 340M | 24 layers, 1024 hidden, 16 heads |
| **RoBERTa** | 125M-355M | Removed NSP, more data, dynamic masking |
| **ALBERT** | 12M-235M | Parameter sharing, factorized embeddings |
| **DistilBERT** | 66M | Distilled, 40% smaller, 60% faster |
| **DeBERTa** | 134M-1.5B | Disentangled attention, enhanced mask decoder |

### BERT for Downstream Tasks

```python
# Classification with BERT
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize input
inputs = tokenizer("Hello, I love using transformers!", return_tensors="pt")

# Forward pass
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

### When to Use Encoder-Only Models

| Task | Why Encoder? |
|------|--------------|
| **Text Classification** | Need full context understanding |
| **Named Entity Recognition** | Bidirectional context crucial |
| **Question Answering (Extractive)** | Find answer spans in context |
| **Semantic Similarity** | Compare sentence meanings |
| **Sentiment Analysis** | Understand overall meaning |

---

## GPT and Decoder-Only Models

### GPT (Generative Pre-trained Transformer)

**Key Innovation:** Autoregressive language modeling - predict next token given previous tokens

### GPT Evolution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPT EVOLUTION                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GPT-1 (2018)                                                               │
│  ├── 117M parameters                                                        │
│  ├── 12 layers                                                              │
│  └── Proved pre-training + fine-tuning works                               │
│                                                                             │
│  GPT-2 (2019)                                                               │
│  ├── 1.5B parameters                                                        │
│  ├── "Too dangerous to release" (initially)                                 │
│  └── Zero-shot task performance emerged                                     │
│                                                                             │
│  GPT-3 (2020)                                                               │
│  ├── 175B parameters                                                        │
│  ├── In-context learning / Few-shot learning                                │
│  └── No fine-tuning needed for many tasks                                   │
│                                                                             │
│  GPT-3.5 / ChatGPT (2022)                                                   │
│  ├── RLHF (Reinforcement Learning from Human Feedback)                      │
│  ├── Instruction following                                                  │
│  └── Conversational ability                                                 │
│                                                                             │
│  GPT-4 (2023)                                                               │
│  ├── Multimodal (text + images)                                             │
│  ├── Significantly improved reasoning                                       │
│  └── Longer context (8K → 32K → 128K tokens)                               │
│                                                                             │
│  GPT-4o (2024)                                                              │
│  ├── "Omni" - native multimodal                                             │
│  ├── Voice, vision, text unified                                            │
│  └── Real-time capabilities                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How GPT Works (Autoregressive Generation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTOREGRESSIVE GENERATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Objective: Predict next token                                     │
│                                                                             │
│  P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁..xₙ₋₁)│
│                                                                             │
│  Loss = -Σ log P(xₜ | x₁, ..., xₜ₋₁)                                        │
│                                                                             │
│  Generation Process:                                                        │
│                                                                             │
│  Input:    "The cat"                                                        │
│              ↓                                                              │
│  Step 1:  "The cat" → Model → P(next) → "sat" (sample)                     │
│              ↓                                                              │
│  Step 2:  "The cat sat" → Model → P(next) → "on" (sample)                  │
│              ↓                                                              │
│  Step 3:  "The cat sat on" → Model → P(next) → "the" (sample)              │
│              ↓                                                              │
│  Step 4:  "The cat sat on the" → Model → P(next) → "mat" (sample)          │
│              ↓                                                              │
│  Output:  "The cat sat on the mat"                                         │
│                                                                             │
│  Causal Masking (during training):                                          │
│                                                                             │
│            The  cat  sat  on                                                │
│     The  [  1    0    0    0  ]                                            │
│     cat  [  1    1    0    0  ]     1 = can attend                         │
│     sat  [  1    1    1    0  ]     0 = cannot attend (masked)             │
│     on   [  1    1    1    1  ]                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decoding Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DECODING STRATEGIES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. GREEDY DECODING                                                         │
│     • Always pick highest probability token                                 │
│     • Fast but can be repetitive                                            │
│     • next_token = argmax(P(x|context))                                     │
│                                                                             │
│  2. BEAM SEARCH                                                             │
│     • Keep top-k sequences at each step                                     │
│     • More diverse but still deterministic                                  │
│     • beam_width = 4-10 typically                                           │
│                                                                             │
│  3. TEMPERATURE SAMPLING                                                    │
│     • P'(x) = softmax(logits / T)                                          │
│     • T < 1: More focused/deterministic                                     │
│     • T > 1: More random/creative                                           │
│     • T = 1: Original distribution                                          │
│                                                                             │
│  4. TOP-K SAMPLING                                                          │
│     • Only sample from top-k most likely tokens                             │
│     • Prevents sampling very unlikely tokens                                │
│     • k = 40-100 typically                                                  │
│                                                                             │
│  5. TOP-P (NUCLEUS) SAMPLING                                                │
│     • Sample from smallest set where cumsum(P) > p                          │
│     • Dynamic vocabulary size                                               │
│     • p = 0.9-0.95 typically                                                │
│                                                                             │
│  6. REPETITION PENALTY                                                      │
│     • Reduce probability of already-generated tokens                        │
│     • Prevents loops and repetition                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Different decoding strategies
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    
    # Temperature sampling
    do_sample=True,
    temperature=0.7,
    
    # Top-k and Top-p
    top_k=50,
    top_p=0.95,
    
    # Repetition penalty
    repetition_penalty=1.2,
    
    # Stopping criteria
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## Modern LLM Architectures

### Key Innovations in Modern LLMs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODERN LLM INNOVATIONS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ROTARY POSITION EMBEDDINGS (RoPE)                                       │
│     • Used in: LLaMA, Mistral, Qwen                                        │
│     • Encodes position through rotation of query/key vectors               │
│     • Better extrapolation to longer sequences                              │
│     • Relative position encoding                                            │
│                                                                             │
│  2. GROUPED QUERY ATTENTION (GQA)                                           │
│     • Used in: LLaMA 2, Mistral                                            │
│     • Shares key-value heads across multiple query heads                    │
│     • Reduces memory and compute while maintaining quality                  │
│     • Middle ground between MHA and MQA                                     │
│                                                                             │
│     MHA:    Q₁ Q₂ Q₃ Q₄  ←→  K₁ K₂ K₃ K₄  ←→  V₁ V₂ V₃ V₄               │
│     GQA:    Q₁ Q₂ Q₃ Q₄  ←→  K₁    K₂     ←→  V₁    V₂                    │
│     MQA:    Q₁ Q₂ Q₃ Q₄  ←→  K₁            ←→  V₁                          │
│                                                                             │
│  3. SLIDING WINDOW ATTENTION (SWA)                                          │
│     • Used in: Mistral, Longformer                                         │
│     • Each token attends to fixed window of neighbors                       │
│     • Linear complexity instead of quadratic                                │
│     • Can still capture long-range through stacking                         │
│                                                                             │
│  4. FLASH ATTENTION                                                         │
│     • Memory-efficient attention computation                                │
│     • Avoids materializing full attention matrix                            │
│     • 2-4x speedup, enables longer contexts                                │
│                                                                             │
│  5. MIXTURE OF EXPERTS (MoE)                                                │
│     • Used in: Mixtral, GPT-4 (rumored)                                    │
│     • Multiple "expert" FFN layers                                          │
│     • Router selects top-k experts per token                                │
│     • More parameters, same compute                                         │
│                                                                             │
│     Input → Router → [Expert 1] [Expert 2] ... [Expert N] → Weighted Sum   │
│                         ↑                                                   │
│                    Only top-k activated                                     │
│                                                                             │
│  6. RING ATTENTION                                                          │
│     • Distributes attention across devices                                  │
│     • Enables million+ token contexts                                       │
│     • Used in: Gemini 1.5                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLaMA Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLaMA ARCHITECTURE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Key Features:                                                              │
│  • Pre-normalization (RMSNorm before attention and FFN)                    │
│  • SwiGLU activation function                                               │
│  • Rotary Position Embeddings (RoPE)                                        │
│  • No bias terms in linear layers                                           │
│                                                                             │
│  LLaMA Block:                                                               │
│                                                                             │
│     Input x                                                                 │
│        │                                                                    │
│        ├──────────────────────────────┐                                    │
│        ↓                              │                                    │
│   [RMSNorm]                           │                                    │
│        ↓                              │                                    │
│   [Self-Attention with RoPE]          │ (Residual)                         │
│        ↓                              │                                    │
│        + ←────────────────────────────┘                                    │
│        │                                                                    │
│        ├──────────────────────────────┐                                    │
│        ↓                              │                                    │
│   [RMSNorm]                           │                                    │
│        ↓                              │                                    │
│   [SwiGLU FFN]                        │ (Residual)                         │
│        ↓                              │                                    │
│        + ←────────────────────────────┘                                    │
│        │                                                                    │
│     Output                                                                  │
│                                                                             │
│  SwiGLU:                                                                    │
│     FFN(x) = (Swish(xW₁) ⊙ xW₃) W₂                                        │
│     Swish(x) = x × σ(x)                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Size Comparison

| Model | Parameters | Context | Released |
|-------|-----------|---------|----------|
| GPT-3 | 175B | 4K | 2020 |
| GPT-4 | ~1.8T (MoE) | 128K | 2023 |
| Claude 3 Opus | ~200B | 200K | 2024 |
| LLaMA 2 | 7B-70B | 4K | 2023 |
| LLaMA 3 | 8B-70B | 8K | 2024 |
| Mistral 7B | 7B | 32K | 2023 |
| Mixtral 8x7B | 47B (12B active) | 32K | 2023 |
| Gemini 1.5 Pro | Unknown | 1M+ | 2024 |

---

## Fine-Tuning Techniques

### The Fine-Tuning Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING APPROACHES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FULL FINE-TUNING                                                           │
│  ├── Update ALL model parameters                                            │
│  ├── Requires lots of GPU memory                                            │
│  ├── Risk of catastrophic forgetting                                        │
│  └── Best quality but most expensive                                        │
│                                                                             │
│  PARAMETER-EFFICIENT FINE-TUNING (PEFT)                                     │
│  ├── Only update small subset of parameters                                 │
│  ├── Much less memory required                                              │
│  ├── Faster training                                                        │
│  └── Multiple adapters for different tasks                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LoRA (Low-Rank Adaptation)

**Key Idea:** Instead of updating full weight matrices, add low-rank decomposition.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LoRA                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Original: h = Wx                                                           │
│                                                                             │
│  LoRA:     h = Wx + BAx                                                     │
│                                                                             │
│  Where:                                                                     │
│  • W ∈ ℝ^(d×d) is frozen                                                   │
│  • A ∈ ℝ^(r×d) - down projection (r << d)                                  │
│  • B ∈ ℝ^(d×r) - up projection                                             │
│  • r = rank (typically 4-64)                                                │
│                                                                             │
│            ┌─────────────────────────────────┐                              │
│            │                                 │                              │
│     x ────→│  W (frozen, d×d)               │────┐                         │
│            │                                 │    │                         │
│            └─────────────────────────────────┘    │                         │
│                                                   + ──→ h                   │
│            ┌─────────────────────────────────┐    │                         │
│            │                                 │    │                         │
│     x ────→│  A (r×d) → B (d×r)             │────┘                         │
│            │  (trainable, low-rank)          │                              │
│            └─────────────────────────────────┘                              │
│                                                                             │
│  Benefits:                                                                  │
│  • Original parameters: d² = 4096² = 16.7M                                 │
│  • LoRA parameters: 2 × d × r = 2 × 4096 × 8 = 65K                        │
│  • 99.6% reduction in trainable parameters!                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                       # Rank
    lora_alpha=32,             # Scaling factor
    lora_dropout=0.1,          # Dropout
    target_modules=[           # Which layers to adapt
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
    ],
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

### QLoRA (Quantized LoRA)

**Key Innovation:** Combine 4-bit quantization with LoRA for even more efficiency.

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Then apply LoRA as before
peft_model = get_peft_model(model, lora_config)
```

### Other PEFT Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **LoRA** | Low-rank weight updates | General fine-tuning |
| **QLoRA** | LoRA + 4-bit quantization | Limited GPU memory |
| **Prefix Tuning** | Learnable prefix tokens | Generation tasks |
| **Prompt Tuning** | Soft prompts | Simple adaptation |
| **Adapter** | Bottleneck modules | Multiple tasks |
| **IA3** | Learned vectors scale activations | Very efficient |

### Instruction Fine-Tuning

**Goal:** Teach model to follow instructions rather than just predict next token.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INSTRUCTION FINE-TUNING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Data Format:                                                      │
│                                                                             │
│  {                                                                          │
│    "instruction": "Summarize the following article",                        │
│    "input": "The article text goes here...",                                │
│    "output": "This is the summary..."                                       │
│  }                                                                          │
│                                                                             │
│  Prompt Template:                                                           │
│                                                                             │
│  ### Instruction:                                                           │
│  {instruction}                                                              │
│                                                                             │
│  ### Input:                                                                 │
│  {input}                                                                    │
│                                                                             │
│  ### Response:                                                              │
│  {output}                                                                   │
│                                                                             │
│  Popular Datasets:                                                          │
│  • Alpaca (52K instructions)                                                │
│  • Dolly (15K instructions)                                                 │
│  • OpenAssistant (160K conversations)                                       │
│  • ShareGPT (90K conversations)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RLHF (Reinforcement Learning from Human Feedback)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RLHF PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: Supervised Fine-Tuning (SFT)                                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Base Model ──→ Fine-tune on demonstrations ──→ SFT Model       │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  STEP 2: Reward Model Training                                              │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Prompt → SFT Model → Multiple Responses                        │       │
│  │                           ↓                                     │       │
│  │              Human ranks responses: A > B > C > D               │       │
│  │                           ↓                                     │       │
│  │              Train Reward Model to predict rankings             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  STEP 3: PPO (Proximal Policy Optimization)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                 │       │
│  │  Prompt → Policy Model → Response → Reward Model → Score        │       │
│  │              ↑                                       │          │       │
│  │              └───────── Update with PPO ─────────────┘          │       │
│  │                                                                 │       │
│  │  Objective: Maximize reward while staying close to SFT model    │       │
│  │                                                                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Modern Alternatives:                                                       │
│  • DPO (Direct Preference Optimization) - No reward model needed           │
│  • ORPO (Odds Ratio Preference Optimization)                               │
│  • KTO (Kahneman-Tversky Optimization)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DPO (Direct Preference Optimization)

**Key Innovation:** Skip reward model, directly optimize from preferences.

```python
# DPO Loss (simplified)
loss = -log(σ(β × (log π(y_w|x) - log π_ref(y_w|x)) 
              - β × (log π(y_l|x) - log π_ref(y_l|x))))

# Where:
# y_w = preferred response
# y_l = dispreferred response
# π = policy model
# π_ref = reference (SFT) model
# β = temperature parameter
```

---

## Prompt Engineering

### What is Prompt Engineering?

The art and science of crafting inputs to get optimal outputs from LLMs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROMPT ENGINEERING TECHNIQUES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ZERO-SHOT PROMPTING                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Prompt: "Translate to French: Hello, how are you?"             │    │
│     │  Output: "Bonjour, comment allez-vous?"                         │    │
│     │                                                                 │    │
│     │  • No examples provided                                         │    │
│     │  • Relies on model's pre-trained knowledge                      │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. FEW-SHOT PROMPTING                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Prompt:                                                        │    │
│     │  "Classify sentiment:                                           │    │
│     │   'I love this!' → Positive                                    │    │
│     │   'This is terrible.' → Negative                               │    │
│     │   'Amazing product!' → "                                       │    │
│     │  Output: "Positive"                                            │    │
│     │                                                                 │    │
│     │  • Provide examples to guide the model                          │    │
│     │  • In-context learning                                          │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. CHAIN-OF-THOUGHT (CoT)                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Prompt: "Q: If John has 3 apples and buys 2 more, then gives   │    │
│     │  away 1, how many does he have?                                 │    │
│     │  Let's think step by step."                                     │    │
│     │                                                                 │    │
│     │  Output: "Step 1: John starts with 3 apples.                    │    │
│     │          Step 2: He buys 2 more: 3 + 2 = 5 apples.             │    │
│     │          Step 3: He gives away 1: 5 - 1 = 4 apples.            │    │
│     │          Answer: 4 apples"                                      │    │
│     │                                                                 │    │
│     │  • Encourages step-by-step reasoning                            │    │
│     │  • Significantly improves math/logic tasks                      │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  4. TREE OF THOUGHTS (ToT)                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  • Explore multiple reasoning paths                             │    │
│     │  • Evaluate and backtrack if needed                             │    │
│     │  • Good for complex problem-solving                             │    │
│     │                                                                 │    │
│     │           Problem                                               │    │
│     │              │                                                  │    │
│     │     ┌───────┼───────┐                                          │    │
│     │     ↓       ↓       ↓                                          │    │
│     │  Path A  Path B  Path C                                        │    │
│     │     │       │       │                                          │    │
│     │  Evaluate each, continue best                                   │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  5. SELF-CONSISTENCY                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  • Sample multiple reasoning paths                              │    │
│     │  • Take majority vote on final answer                           │    │
│     │  • Reduces random errors                                        │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  6. ReAct (Reasoning + Acting)                                              │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Thought: I need to find the current weather in Paris.          │    │
│     │  Action: search("Paris weather today")                          │    │
│     │  Observation: [Search results...]                               │    │
│     │  Thought: The weather is 15°C and cloudy.                       │    │
│     │  Answer: It's currently 15°C and cloudy in Paris.              │    │
│     │                                                                 │    │
│     │  • Interleave reasoning with tool use                           │    │
│     │  • Foundation for AI agents                                     │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prompt Template Best Practices

```python
# Good prompt structure
prompt_template = """
You are a {role} expert.

Context: {context}

Task: {task}

Requirements:
- {requirement_1}
- {requirement_2}
- {requirement_3}

Format your response as:
{output_format}

Input: {input}

Output:
"""

# Example
prompt = prompt_template.format(
    role="Python programming",
    context="Building a web scraping application",
    task="Write a function to extract all links from a webpage",
    requirement_1="Use the requests and BeautifulSoup libraries",
    requirement_2="Handle errors gracefully",
    requirement_3="Return a list of URLs",
    output_format="Python code with comments",
    input="https://example.com"
)
```

### System Prompts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PROMPT STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ROLE DEFINITION                                                            │
│  "You are an expert data scientist with 10 years of experience..."         │
│                                                                             │
│  BEHAVIORAL CONSTRAINTS                                                     │
│  "Always provide accurate information. If unsure, say so..."               │
│                                                                             │
│  OUTPUT FORMAT                                                              │
│  "Respond in JSON format with keys: 'answer', 'confidence', 'sources'"     │
│                                                                             │
│  EXAMPLES (optional)                                                        │
│  "Here's an example of how to respond:..."                                 │
│                                                                             │
│  SAFETY GUIDELINES                                                          │
│  "Do not provide harmful information. Refuse inappropriate requests..."    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Augmented Generation (RAG)

### What is RAG?

RAG combines LLMs with external knowledge retrieval to provide accurate, up-to-date, and verifiable responses.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Without RAG:                                                               │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  User Query ──→ LLM ──→ Response (from parametric memory)    │          │
│  │                                                               │          │
│  │  Problems:                                                    │          │
│  │  • Knowledge cutoff (outdated info)                          │          │
│  │  • Hallucinations                                            │          │
│  │  • No source verification                                    │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  With RAG:                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                                                               │          │
│  │  User Query                                                   │          │
│  │      │                                                        │          │
│  │      ↓                                                        │          │
│  │  [Embedding Model] ──→ Query Vector                          │          │
│  │      │                                                        │          │
│  │      ↓                                                        │          │
│  │  [Vector Database] ──→ Retrieve Top-K Similar Documents      │          │
│  │      │                                                        │          │
│  │      ↓                                                        │          │
│  │  [Augmented Prompt] = Query + Retrieved Context              │          │
│  │      │                                                        │          │
│  │      ↓                                                        │          │
│  │  [LLM] ──→ Response (grounded in retrieved docs)             │          │
│  │                                                               │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INDEXING PHASE (Offline):                                                  │
│                                                                             │
│  Documents ──→ Chunking ──→ Embedding ──→ Vector Store                     │
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   PDF       │     │  Chunk 1    │     │  [0.1, 0.5, │                   │
│  │   HTML      │ ──→ │  Chunk 2    │ ──→ │   ...]      │ ──→ Vector DB    │
│  │   TXT       │     │  Chunk 3    │     │  [0.3, 0.2, │                   │
│  │   ...       │     │  ...        │     │   ...]      │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                                             │
│  RETRIEVAL PHASE (Online):                                                  │
│                                                                             │
│  Query ──→ Embed ──→ Search ──→ Retrieve ──→ Rerank ──→ Context           │
│                                                                             │
│  GENERATION PHASE (Online):                                                 │
│                                                                             │
│  Context + Query ──→ Prompt ──→ LLM ──→ Response                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Chunking Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CHUNKING STRATEGIES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FIXED-SIZE CHUNKING                                                     │
│     • Split by character/token count                                        │
│     • Simple but may break semantic units                                   │
│     • chunk_size=512, overlap=50                                           │
│                                                                             │
│  2. SENTENCE-BASED CHUNKING                                                 │
│     • Split at sentence boundaries                                          │
│     • Preserves complete thoughts                                           │
│     • Variable chunk sizes                                                  │
│                                                                             │
│  3. SEMANTIC CHUNKING                                                       │
│     • Use embeddings to find natural break points                           │
│     • Group semantically similar sentences                                  │
│     • Higher quality but more complex                                       │
│                                                                             │
│  4. RECURSIVE CHUNKING                                                      │
│     • Try different separators hierarchically                               │
│     • \n\n → \n → . → space                                                │
│     • Balances structure and size                                           │
│                                                                             │
│  5. DOCUMENT-STRUCTURE CHUNKING                                             │
│     • Use headers, sections, paragraphs                                     │
│     • Best for structured documents                                         │
│     • Preserves document hierarchy                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation with LangChain

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 5. Create RAG chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce", "refine"
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain({"query": "What is the main topic of this document?"})
print(result["result"])
print("Sources:", result["source_documents"])
```

### Advanced RAG Techniques

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED RAG TECHNIQUES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. HYBRID SEARCH                                                           │
│     • Combine dense (embedding) + sparse (BM25) retrieval                   │
│     • Better coverage of keyword and semantic matches                       │
│     • score = α × dense_score + (1-α) × sparse_score                       │
│                                                                             │
│  2. RERANKING                                                               │
│     • Retrieve more docs, rerank with cross-encoder                         │
│     • Cross-encoder: Compare query-doc pairs directly                       │
│     • Better precision at cost of latency                                   │
│                                                                             │
│  3. QUERY TRANSFORMATION                                                    │
│     • HyDE: Generate hypothetical answer, embed that                        │
│     • Query expansion: Generate multiple query variants                     │
│     • Step-back prompting: Ask broader question first                       │
│                                                                             │
│  4. SELF-RAG                                                                │
│     • Model decides when to retrieve                                        │
│     • Critiques its own outputs                                             │
│     • More dynamic retrieval                                                │
│                                                                             │
│  5. CORRECTIVE RAG (CRAG)                                                   │
│     • Evaluate retrieval quality                                            │
│     • If low quality, trigger web search                                    │
│     • Self-correcting pipeline                                              │
│                                                                             │
│  6. MULTI-QUERY RAG                                                         │
│     • Generate multiple queries from user input                             │
│     • Retrieve for each, combine results                                    │
│     • Better recall                                                         │
│                                                                             │
│  7. PARENT DOCUMENT RETRIEVER                                               │
│     • Index small chunks for retrieval                                      │
│     • Return larger parent chunks for context                               │
│     • Balance precision and context                                         │
│                                                                             │
│  8. CONTEXTUAL COMPRESSION                                                  │
│     • Compress retrieved docs to relevant parts                             │
│     • Reduce noise in context                                               │
│     • Fit more information in context window                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vector Databases and Embeddings

### Embedding Models

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING MODELS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  What are Embeddings?                                                       │
│  • Dense vector representations of text                                     │
│  • Capture semantic meaning                                                 │
│  • Similar meanings → similar vectors                                       │
│                                                                             │
│  "king" - "man" + "woman" ≈ "queen"  (classic example)                     │
│                                                                             │
│  Popular Embedding Models:                                                  │
│                                                                             │
│  ┌─────────────────┬────────────┬────────────┬──────────────────────┐      │
│  │ Model           │ Dimensions │ Max Tokens │ Notes                │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ OpenAI text-    │ 1536/3072  │ 8191       │ Best quality, paid   │      │
│  │ embedding-3     │            │            │                      │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ Cohere embed-v3 │ 1024       │ 512        │ Multilingual         │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ BGE-large       │ 1024       │ 512        │ Open source, strong  │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ E5-large        │ 1024       │ 512        │ Microsoft, versatile │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ GTE-large       │ 1024       │ 512        │ Alibaba, efficient   │      │
│  ├─────────────────┼────────────┼────────────┼──────────────────────┤      │
│  │ all-MiniLM-L6   │ 384        │ 256        │ Fast, lightweight    │      │
│  └─────────────────┴────────────┴────────────┴──────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vector Similarity Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMILARITY METRICS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. COSINE SIMILARITY                                                       │
│     • Measures angle between vectors                                        │
│     • Range: [-1, 1] (usually [0, 1] for normalized)                       │
│     • Most common for text embeddings                                       │
│                                                                             │
│     cos(A, B) = (A · B) / (||A|| × ||B||)                                  │
│                                                                             │
│  2. EUCLIDEAN DISTANCE (L2)                                                 │
│     • Straight-line distance                                                │
│     • Range: [0, ∞)                                                        │
│     • Sensitive to magnitude                                                │
│                                                                             │
│     d(A, B) = √Σ(Aᵢ - Bᵢ)²                                                 │
│                                                                             │
│  3. DOT PRODUCT                                                             │
│     • Simple inner product                                                  │
│     • Faster computation                                                    │
│     • Range: (-∞, ∞)                                                       │
│                                                                             │
│     A · B = Σ(Aᵢ × Bᵢ)                                                     │
│                                                                             │
│  4. MANHATTAN DISTANCE (L1)                                                 │
│     • Sum of absolute differences                                           │
│     • More robust to outliers                                               │
│                                                                             │
│     d(A, B) = Σ|Aᵢ - Bᵢ|                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vector Databases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VECTOR DATABASES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┬──────────────────────────────────────────────────┐    │
│  │ Database        │ Key Features                                     │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ Pinecone        │ Fully managed, serverless, high performance      │    │
│  │                 │ Great for production, easy scaling               │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ Weaviate        │ Open source, GraphQL API, hybrid search          │    │
│  │                 │ Good ML integrations                             │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ Milvus          │ Open source, highly scalable                     │    │
│  │                 │ GPU acceleration, enterprise ready               │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ Qdrant          │ Open source, Rust-based, fast                    │    │
│  │                 │ Good filtering, payload storage                  │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ Chroma          │ Open source, embedded, developer-friendly        │    │
│  │                 │ Great for prototyping and local dev              │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ FAISS           │ Facebook library, very fast                      │    │
│  │                 │ Not a full DB, but excellent for search          │    │
│  ├─────────────────┼──────────────────────────────────────────────────┤    │
│  │ pgvector        │ PostgreSQL extension                             │    │
│  │                 │ Use existing Postgres, simpler ops               │    │
│  └─────────────────┴──────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Examples

```python
# Using Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "..."}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"text": "..."}},
])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

# Using ChromaDB (local)
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")

# Add documents
collection.add(
    documents=["doc1 text", "doc2 text"],
    metadatas=[{"source": "a"}, {"source": "b"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

---

## AI Agents and Autonomous Systems

### What are AI Agents?

AI agents are systems that use LLMs as the reasoning engine to autonomously plan and execute tasks, using tools and interacting with external systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI AGENT ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────┐                                 │
│                         │   User Query    │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                         AGENT CORE                                 │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                    LLM (Brain)                               │ │     │
│  │  │  • Understands goals                                         │ │     │
│  │  │  • Plans actions                                             │ │     │
│  │  │  • Reasons about observations                                │ │     │
│  │  │  • Decides next steps                                        │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  │                              │                                    │     │
│  │                              ↓                                    │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                    MEMORY                                    │ │     │
│  │  │  • Short-term: Current conversation                         │ │     │
│  │  │  • Long-term: Vector store of past interactions             │ │     │
│  │  │  • Working: Current task state                              │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                  │                                          │
│                                  ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                         TOOLS                                      │     │
│  │                                                                    │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │     │
│  │  │ Search  │ │  Code   │ │  API    │ │Database │ │  File   │    │     │
│  │  │ Engine  │ │ Executor│ │  Calls  │ │  Query  │ │  System │    │     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │     │
│  │                                                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                  │                                          │
│                                  ↓                                          │
│                         ┌─────────────────┐                                 │
│                         │   Final Output  │                                 │
│                         └─────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Frameworks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT FRAMEWORKS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LANGCHAIN AGENTS                                                        │
│     • Flexible tool use                                                     │
│     • Many pre-built tools                                                  │
│     • ReAct, OpenAI Functions, etc.                                        │
│                                                                             │
│  2. AUTOGEN (Microsoft)                                                     │
│     • Multi-agent conversations                                             │
│     • Agent collaboration                                                   │
│     • Code execution                                                        │
│                                                                             │
│  3. CREWAI                                                                  │
│     • Role-based agents                                                     │
│     • Process orchestration                                                 │
│     • Easy to define crews                                                  │
│                                                                             │
│  4. OPENAI ASSISTANTS API                                                   │
│     • Managed agent infrastructure                                          │
│     • Built-in tools (code, retrieval)                                      │
│     • Stateful threads                                                      │
│                                                                             │
│  5. LLAMAINDEX AGENTS                                                       │
│     • Data-focused agents                                                   │
│     • Strong RAG integration                                                │
│     • Query planning                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ReAct Pattern Implementation

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Define tools
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Search results for: {query}"

def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

tools = [
    Tool(
        name="Search",
        func=search_web,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Perform mathematical calculations"
    ),
]

# Create agent
llm = ChatOpenAI(model="gpt-4")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)

# Run agent
result = agent_executor.invoke({
    "input": "What is the population of France and what is it divided by 1000?"
})
```

### Multi-Agent Systems

```python
# Using CrewAI
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI',
    backstory='You are an expert at analyzing trends...',
    tools=[search_tool, scrape_tool],
    llm=llm
)

writer = Agent(
    role='Tech Content Writer',
    goal='Write engaging content about AI discoveries',
    backstory='You are a renowned tech writer...',
    tools=[],
    llm=llm
)

# Define tasks
research_task = Task(
    description='Research the latest AI trends...',
    expected_output='A comprehensive report...',
    agent=researcher
)

writing_task = Task(
    description='Write a blog post based on the research...',
    expected_output='A polished blog post...',
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

# Execute
result = crew.kickoff()
```

### Function Calling

```python
# OpenAI Function Calling
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Execute the function
    result = get_weather(**arguments)
    
    # Send result back to model
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })
    
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

---

## Generative AI

### Types of Generative AI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GENERATIVE AI LANDSCAPE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEXT GENERATION                                                            │
│  ├── Large Language Models (GPT-4, Claude, etc.)                           │
│  ├── Code Generation (Codex, GitHub Copilot)                               │
│  └── Creative Writing (stories, poetry, scripts)                           │
│                                                                             │
│  IMAGE GENERATION                                                           │
│  ├── Diffusion Models (Stable Diffusion, DALL-E 3, Midjourney)            │
│  ├── GANs (StyleGAN, BigGAN)                                               │
│  └── Image Editing (inpainting, outpainting)                               │
│                                                                             │
│  VIDEO GENERATION                                                           │
│  ├── Text-to-Video (Sora, Runway, Pika)                                    │
│  ├── Video Editing (frame interpolation)                                   │
│  └── Animation                                                              │
│                                                                             │
│  AUDIO GENERATION                                                           │
│  ├── Text-to-Speech (ElevenLabs, Bark)                                     │
│  ├── Music Generation (Suno, Udio)                                         │
│  └── Voice Cloning                                                          │
│                                                                             │
│  3D GENERATION                                                              │
│  ├── Text-to-3D (Point-E, Shap-E)                                          │
│  └── NeRF (Neural Radiance Fields)                                         │
│                                                                             │
│  MULTIMODAL                                                                 │
│  ├── GPT-4V (text + images)                                                │
│  ├── Gemini (text + images + video + audio)                                │
│  └── Any-to-Any models                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diffusion Models

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DIFFUSION MODELS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORWARD PROCESS (Training):                                                │
│  Gradually add noise to image until pure noise                              │
│                                                                             │
│  x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ xₜ                                          │
│  Image   +noise  +noise       Pure noise                                    │
│                                                                             │
│  REVERSE PROCESS (Generation):                                              │
│  Learn to denoise step by step                                              │
│                                                                             │
│  xₜ ──→ xₜ₋₁ ──→ ... ──→ x₁ ──→ x₀                                        │
│  Noise  -noise            -noise  Image                                     │
│                                                                             │
│  KEY COMPONENTS:                                                            │
│                                                                             │
│  1. U-Net: Predicts noise to remove at each step                           │
│  2. Text Encoder: CLIP encodes text prompts                                 │
│  3. VAE: Compress/decompress to latent space                               │
│  4. Scheduler: Controls noise addition/removal                              │
│                                                                             │
│  STABLE DIFFUSION ARCHITECTURE:                                             │
│                                                                             │
│  Text Prompt ──→ [CLIP Text Encoder] ──→ Text Embeddings                   │
│                                              │                              │
│                                              ↓                              │
│  Random Noise ──→ [U-Net] ←── Conditioning ──┘                             │
│                      │                                                      │
│                      ↓ (iterate T steps)                                    │
│               Denoised Latent                                               │
│                      │                                                      │
│                      ↓                                                      │
│               [VAE Decoder]                                                 │
│                      │                                                      │
│                      ↓                                                      │
│               Generated Image                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Example

```python
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A futuristic city at sunset, cyberpunk style, highly detailed"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    width=768,
    height=768
).images[0]

image.save("generated_image.png")
```

---

## Vision Transformers (ViT)

### From CNNs to Transformers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VISION TRANSFORMER (ViT)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Traditional CNN:                                                           │
│  • Local receptive fields (convolutions)                                    │
│  • Translation equivariance                                                 │
│  • Hierarchical features                                                    │
│                                                                             │
│  Vision Transformer:                                                        │
│  • Global attention from the start                                          │
│  • Treats image as sequence of patches                                      │
│  • Same architecture as NLP transformers                                    │
│                                                                             │
│  ViT ARCHITECTURE:                                                          │
│                                                                             │
│  Input Image (224×224)                                                      │
│        │                                                                    │
│        ↓ Split into patches (16×16)                                        │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ [P1] [P2] [P3] ... [P196]  (14×14 = 196 patches)   │                   │
│  └─────────────────────────────────────────────────────┘                   │
│        │                                                                    │
│        ↓ Flatten and project                                               │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ [CLS] [E1] [E2] [E3] ... [E196] + Position Emb     │                   │
│  └─────────────────────────────────────────────────────┘                   │
│        │                                                                    │
│        ↓ Transformer Encoder (L layers)                                    │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Multi-Head Self-Attention                           │                   │
│  │ MLP                                                  │                   │
│  │ Layer Norm + Residuals                              │                   │
│  └─────────────────────────────────────────────────────┘                   │
│        │                                                                    │
│        ↓ Use [CLS] token                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Classification Head                                  │                   │
│  └─────────────────────────────────────────────────────┘                   │
│        │                                                                    │
│        ↓                                                                    │
│  Class Prediction                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Modern Vision Models

| Model | Key Innovation |
|-------|----------------|
| **ViT** | First pure transformer for vision |
| **DeiT** | Data-efficient training, distillation |
| **Swin Transformer** | Shifted windows, hierarchical |
| **BEiT** | BERT-style pre-training for images |
| **MAE** | Masked autoencoder, efficient pre-training |
| **CLIP** | Contrastive image-text pre-training |
| **DINO** | Self-supervised vision transformer |
| **SAM** | Segment Anything Model |

---

## Multimodal Models

### The Multimodal Revolution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL ARCHITECTURES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EARLY FUSION:                                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Image ──→ [Image Encoder] ──┐                              │           │
│  │                               ├──→ [Unified Model] ──→ Output│          │
│  │  Text  ──→ [Text Encoder]  ──┘                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  LATE FUSION:                                                               │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Image ──→ [Image Model] ──→ Features ──┐                   │           │
│  │                                          ├──→ Combine ──→ Out│          │
│  │  Text  ──→ [Text Model]  ──→ Features ──┘                   │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  CROSS-ATTENTION (LLaVA-style):                                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  Image ──→ [Vision Encoder] ──→ Image Tokens                │           │
│  │                                      │                       │           │
│  │                                      ↓                       │           │
│  │  Text  ──→ [Tokenizer] ──→ Text Tokens + Image Tokens       │           │
│  │                                      │                       │           │
│  │                                      ↓                       │           │
│  │                               [LLM Decoder]                  │           │
│  │                                      │                       │           │
│  │                                      ↓                       │           │
│  │                               Response                       │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CLIP (Contrastive Language-Image Pre-training)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIP                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training:                                                                  │
│  • 400M image-text pairs from internet                                      │
│  • Contrastive learning: match images with captions                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                                                             │           │
│  │    Image ──→ [Image Encoder] ──→ Image Embedding            │           │
│  │                                        ↓                    │           │
│  │                              Maximize similarity            │           │
│  │                              for matching pairs             │           │
│  │                                        ↑                    │           │
│  │    Text  ──→ [Text Encoder]  ──→ Text Embedding             │           │
│  │                                                             │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  Capabilities:                                                              │
│  • Zero-shot image classification                                           │
│  • Image-text retrieval                                                     │
│  • Foundation for other models (Stable Diffusion)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Major Multimodal Models

| Model | Modalities | Key Features |
|-------|-----------|--------------|
| **GPT-4V** | Text, Images | Strong reasoning, OCR |
| **Gemini** | Text, Images, Video, Audio | Native multimodal |
| **Claude 3** | Text, Images | Long context, safety |
| **LLaVA** | Text, Images | Open source, efficient |
| **Qwen-VL** | Text, Images | Multilingual |
| **DALL-E 3** | Text → Images | High quality generation |
| **Sora** | Text → Video | Long coherent videos |

---

## MLOps and Production Systems

### MLOps Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLOPS LIFECYCLE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│  │  Data   │ ──→ │  Model  │ ──→ │  Model  │ ──→ │  Model  │              │
│  │Ingestion│     │Training │     │Evaluation│     │Deployment│             │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘              │
│       ↑                                               │                    │
│       │                                               │                    │
│       │         ┌─────────────────────────┐          │                    │
│       └─────────│     Monitoring &        │←─────────┘                    │
│                 │     Retraining          │                                │
│                 └─────────────────────────┘                                │
│                                                                             │
│  KEY COMPONENTS:                                                            │
│                                                                             │
│  • Data Versioning: DVC, Delta Lake  



king Strategies

| Strategy | Description |
|----------|-------------|
| **Fixed-Size** | Split by character/token count (chunk_size=512) |
| **Sentence-Based** | Split at sentence boundaries |
| **Semantic** | Use embeddings to find natural break points |
| **Recursive** | Try different separators hierarchically |
| **Document-Structure** | Use headers, sections, paragraphs |

### Implementation with LangChain

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. Create RAG chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain({"query": "What is the main topic?"})
print(result["result"])
```

### Advanced RAG Techniques

| Technique | Description |
|-----------|-------------|
| **Hybrid Search** | Combine dense + sparse (BM25) retrieval |
| **Reranking** | Use cross-encoder to rerank results |
| **Query Transformation** | HyDE, query expansion |
| **Self-RAG** | Model decides when to retrieve |
| **CRAG** | Evaluate and correct retrieval quality |
| **Multi-Query** | Generate multiple queries, combine results |

---

## Vector Databases and Embeddings

### Embedding Models

| Model | Dimensions | Max Tokens | Notes |
|-------|------------|------------|-------|
| OpenAI text-embedding-3 | 1536/3072 | 8191 | Best quality, paid |
| Cohere embed-v3 | 1024 | 512 | Multilingual |
| BGE-large | 1024 | 512 | Open source, strong |
| E5-large | 1024 | 512 | Microsoft, versatile |
| all-MiniLM-L6 | 384 | 256 | Fast, lightweight |

### Similarity Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Cosine** | (A·B)/(‖A‖×‖B‖) | Text embeddings |
| **Euclidean** | √Σ(Aᵢ-Bᵢ)² | When magnitude matters |
| **Dot Product** | Σ(Aᵢ×Bᵢ) | Normalized vectors |

### Vector Databases

| Database | Best For |
|----------|----------|
| **Pinecone** | Managed, production, scale |
| **Weaviate** | Hybrid search, GraphQL |
| **Qdrant** | Fast, Rust-based, self-hosted |
| **Milvus** | Enterprise, GPU support |
| **Chroma** | Development, embedded |
| **pgvector** | Postgres users, simple ops |

---

## AI Agents and Autonomous Systems

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI AGENT ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────┐                                 │
│                         │   User Query    │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                         AGENT CORE                                 │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                    LLM (Brain)                               │ │     │
│  │  │  • Understands goals                                         │ │     │
│  │  │  • Plans actions                                             │ │     │
│  │  │  • Reasons about observations                                │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  │                              │                                    │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                    MEMORY                                    │ │     │
│  │  │  • Short-term: Current conversation                         │ │     │
│  │  │  • Long-term: Vector store of past interactions             │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                  │                                          │
│                                  ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                         TOOLS                                      │     │
│  │  [Search] [Code Executor] [API Calls] [Database] [File System]    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Frameworks

| Framework | Best For |
|-----------|----------|
| **LangChain** | Complex chains, agents, RAG |
| **AutoGen** | Multi-agent collaboration |
| **CrewAI** | Role-based agents |
| **OpenAI Assistants** | Managed infrastructure |
| **LlamaIndex** | Data-focused agents |

### Function Calling Example

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
```

---

## Generative AI

### Types of Generative AI

| Type | Examples | Use Cases |
|------|----------|-----------|
| **Text** | GPT-4, Claude | Chatbots, writing, code |
| **Image** | DALL-E 3, Midjourney, Stable Diffusion | Art, design, marketing |
| **Video** | Sora, Runway, Pika | Film, advertising |
| **Audio** | ElevenLabs, Suno | TTS, music |
| **3D** | Point-E, Shap-E | Gaming, VR |

### Diffusion Models

```
FORWARD PROCESS (Training):
x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ xₜ
Image   +noise  +noise       Pure noise

REVERSE PROCESS (Generation):
xₜ ──→ xₜ₋₁ ──→ ... ──→ x₁ ──→ x₀
Noise  -noise            -noise  Image
```

### Stable Diffusion Components

1. **U-Net**: Predicts noise to remove at each step
2. **Text Encoder**: CLIP encodes text prompts
3. **VAE**: Compress/decompress to latent space
4. **Scheduler**: Controls noise addition/removal

---

## Vision Transformers (ViT)

### ViT Architecture

```
Input Image (224×224)
       │
       ↓ Split into patches (16×16)
┌─────────────────────────────────────────────────────┐
│ [P1] [P2] [P3] ... [P196]  (14×14 = 196 patches)   │
└─────────────────────────────────────────────────────┘
       │
       ↓ Flatten and project
┌─────────────────────────────────────────────────────┐
│ [CLS] [E1] [E2] [E3] ... [E196] + Position Emb     │
└─────────────────────────────────────────────────────┘
       │
       ↓ Transformer Encoder (L layers)
       │
       ↓ Use [CLS] token
       │
Class Prediction
```

### Modern Vision Models

| Model | Key Innovation |
|-------|----------------|
| **ViT** | First pure transformer for vision |
| **DeiT** | Data-efficient training |
| **Swin** | Shifted windows, hierarchical |
| **CLIP** | Contrastive image-text pre-training |
| **SAM** | Segment Anything Model |

---

## Multimodal Models

### Major Multimodal Models

| Model | Modalities | Key Features |
|-------|-----------|--------------|
| **GPT-4V** | Text, Images | Strong reasoning, OCR |
| **Gemini** | Text, Images, Video, Audio | Native multimodal |
| **Claude 3** | Text, Images | Long context, safety |
| **LLaVA** | Text, Images | Open source |
| **DALL-E 3** | Text → Images | High quality |
| **Sora** | Text → Video | Long coherent videos |

---

## MLOps and Production Systems

### MLOps Lifecycle

```
Data Ingestion → Model Training → Model Evaluation → Model Deployment
       ↑                                                    │
       └──────────── Monitoring & Retraining ←──────────────┘
```

### Key Tools

| Category | Tools |
|----------|-------|
| **Experiment Tracking** | MLflow, Weights & Biases, Neptune |
| **Model Registry** | MLflow, SageMaker, Vertex AI |
| **Orchestration** | Airflow, Kubeflow, Prefect |
| **Serving** | vLLM, TGI, Triton |
| **Monitoring** | Prometheus, Grafana, Evidently |

### FastAPI Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(request: GenerationRequest):
    # Model inference here
    return {"generated_text": "..."}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## Model Optimization and Deployment

### Quantization

| Precision | Bits | Memory | Speed |
|-----------|------|--------|-------|
| FP32 | 32 | Baseline | Baseline |
| FP16 | 16 | 50% | ~2x |
| INT8 | 8 | 25% | ~4x |
| INT4 | 4 | 12.5% | ~8x |

### Quantization Methods

| Method | Description |
|--------|-------------|
| **PTQ** | Post-training quantization |
| **QAT** | Quantization-aware training |
| **GPTQ** | Layer-wise for LLMs |
| **AWQ** | Activation-aware weight quantization |
| **GGUF** | llama.cpp format, CPU-optimized |

### Inference Engines

| Engine | Best For |
|--------|----------|
| **vLLM** | High-throughput LLM serving |
| **TensorRT-LLM** | NVIDIA GPUs |
| **llama.cpp** | CPU inference |
| **Ollama** | Local deployment |
| **TGI** | HuggingFace production |

---

## Evaluation and Benchmarking

### LLM Benchmarks

| Benchmark | What it Measures |
|-----------|------------------|
| MMLU | Multi-task language understanding |
| HellaSwag | Commonsense reasoning |
| HumanEval | Code generation |
| GSM8K | Grade school math |
| TruthfulQA | Truthfulness |
| MT-Bench | Multi-turn conversation |

### RAG Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Answer matches context? |
| **Answer Relevancy** | Addresses question? |
| **Context Precision** | Retrieved context relevant? |
| **Groundedness** | Claims supported by context? |

---

## Safety, Alignment, and Ethics

### Key Concerns

| Concern | Mitigation |
|---------|------------|
| **Hallucinations** | RAG, grounding |
| **Harmful Content** | Content filtering, RLHF |
| **Bias** | Diverse data, audits |
| **Privacy** | Differential privacy |
| **Prompt Injection** | Input validation, guardrails |

### Alignment Techniques

- **RLHF**: Learn from human preferences
- **Constitutional AI**: Self-critique against principles
- **DPO**: Direct preference optimization
- **Red Teaming**: Adversarial testing

---

## Current Industry Tools and Frameworks

### LLM Frameworks

| Framework | Best For |
|-----------|----------|
| LangChain | Complex chains, RAG |
| LlamaIndex | Data indexing, RAG |
| Haystack | Search, QA |
| DSPy | Programmatic prompting |

### Cloud Platforms

| Platform | Key Services |
|----------|--------------|
| AWS Bedrock | Multiple LLMs, RAG |
| Azure OpenAI | GPT models, enterprise |
| Google Vertex | Gemini, PaLM |
| Hugging Face | Model hub, inference |

---

## Future Trends (2024-2025)

1. **Smaller, More Efficient Models** - Phi-3, Mistral showing small can be powerful
2. **Multimodal Everything** - Native multimodal training
3. **Longer Context** - 1M+ tokens becoming standard
4. **Better Reasoning** - Chain-of-thought improvements
5. **Autonomous Agents** - More reliable tool use
6. **On-Device AI** - Privacy-preserving local models
7. **New Architectures** - State Space Models (Mamba)

---

## Quick Reference

### Model Selection Guide

| Task | Recommended Models |
|------|-------------------|
| Chat/Assistant | GPT-4, Claude 3, Gemini |
| Code | GPT-4, Claude 3, DeepSeek Coder |
| Classification | BERT, RoBERTa, DeBERTa |
| Embeddings | OpenAI Ada, BGE, E5 |
| Image Gen | DALL-E 3, Midjourney, SD |
| Self-Hosted | LLaMA 3, Mistral, Qwen |

### Essential Libraries

```bash
# Core
pip install torch transformers datasets

# LLM Apps
pip install langchain llama-index openai

# Vector DBs
pip install chromadb pinecone-client

# Fine-tuning
pip install peft trl bitsandbytes

# Serving
pip install vllm fastapi
```

---

## Conclusion

This guide covered modern AI from Transformers to production systems:

1. **Transformers are the foundation** - Understanding attention is crucial
2. **LLMs revolutionized NLP** - GPT, Claude, LLaMA families
3. **Fine-tuning is accessible** - LoRA/QLoRA on consumer hardware
4. **RAG solves knowledge limits** - Combine retrieval with generation
5. **Agents are the future** - LLMs as reasoning engines
6. **Optimization matters** - Quantization for production
7. **Safety is essential** - Guardrails and alignment

> **"The goal is not to replace human intelligence, but to augment it."**

---

*This guide represents the state of the art as of 2024. The field evolves rapidly!*
