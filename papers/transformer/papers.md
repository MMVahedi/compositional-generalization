# Papers
## **2024**

### [Inducing Systematicity in Transformers by Attending to Structurally Quantized Embeddings](https://aclanthology.org/2024.acl-long.455.pdf)

- **Year:** 2024
- **Conference/Venue:** ACL 2024 – Long Papers
- **Institutions:** University of North Carolina at Chapel Hill
- **Abstract:** Proposes the **SQ-Transformer**, which integrates _structure-oriented vector quantization_ (SoVQ) to cluster embeddings by structural role and adds regularization on attention heads for structural invariance. The approach yields stronger systematic generalization on semantic parsing and translation benchmarks.
- **Keywords:** systematicity, transformer, vector quantization, structural embedding, compositional generalization

### [Latent Plan Transformer for Trajectory Abstraction: Planning as Latent Space Inference](https://proceedings.neurips.cc/paper_files/paper/2024/file/df22a19686a558e74f038e6277a51f68-Paper-Conference.pdf)

- **Year:** 2024    
- **Conference/Venue:** NeurIPS 2024
- **Institutions:** UCLA; Amazon AI; UC Irvine
- **Abstract:** Introduces the **Latent Plan Transformer (LPT)**, which formulates trajectory planning as _latent-space inference_. The latent plan conditions a Transformer trajectory generator, enabling effective offline RL planning from sub-optimal trajectories.
- **Keywords:** transformer, latent variable, trajectory abstraction, planning, offline reinforcement learning

### [Out-of-Distribution Generalization via Composition: A Lens Through Induction Heads in Transformers](https://arxiv.org/pdf/2408.09503v2.pdf)

- **Year:** 2024
- **Conference/Venue:** To be presented at ICLR 2024
- **Institutions:** National Key Laboratory of General Artificial Intelligence, BIGAI, Beijing; Department of Statistics, University of Wisconsin–Madison
- **Abstract:** This work examines **out-of-distribution (OOD) generalization** in large language models (LLMs) by analyzing their ability to infer hidden rules from input prompts. The study shows how **induction heads** (IHs) in Transformers aid in achieving OOD generalization through compositional structures. The **common bridge representation hypothesis** is introduced, highlighting the role of shared latent subspaces in this process.
- **Keywords:** OOD generalization, induction heads, Transformers, compositionality, common bridge representation.

### [Benchmarking and Improving Compositional Generalization of Multi-aspect Controllable Text Generation](https://aclanthology.org/2024.acl-long.351.pdf)

- **Year:** 2024    
- **Conference/Venue:** ACL 2024 – Long Papers
- **Institutions:** Various NLP research groups
- **Abstract:** Establishes new benchmarks for **multi-aspect controllable text generation**, examining generalization to unseen combinations of control attributes. Proposes training strategies that enhance compositional control in generated text.
- **Keywords:** compositional generalization, controllable generation, multi-aspect text control, benchmarking

### [Strengthening Structural Inductive Biases by Pre-training to Perform Syntactic Transformations](https://arxiv.org/pdf/2407.04543)

- **Year:** 2024
- **Conference/Venue:** Pre-print on arXiv
- **Institutions:** (Unspecified)
- **Abstract:** Shows that **pre-training on syntactic transformations** (e.g., parse tree rewriting) enhances structural inductive biases and yields better systematic generalization on syntax-sensitive benchmarks.
- **Keywords:** inductive bias, pre-training, syntax, compositional generalization, structure learning

### [LEARNING SYNTAX WITHOUT PLANTING TREES: UNDERSTANDING HIERARCHICAL GENERALIZATION IN TRANSFORMERS](https://arxiv.org/pdf/2404.16367v3)

- **Year:** 2024
- **Conference/Venue:** arXiv (v3)
- **Institutions:** (Unspecified)
- **Abstract:** Analyzes how Transformers acquire **hierarchical syntactic representations** without explicit tree supervision. Proposes evaluation methods for hierarchical generalization and reveals emergent tree-like organization in self-attention patterns.
- **Keywords:** hierarchical generalization, syntax, transformer, compositionality, interpretability    

### [Not Just Object, But State: Compositional Incremental Learning without Forgetting](https://openreview.net/pdf?id=2LRZhbTDtA)

- **Year:** 2024
- **Conference/Venue:** OpenReview (pre-publication submission)
- **Institutions:** (Unspecified)    
- **Abstract:** Introduces **compositional incremental learning**, where models learn not only new objects but also new **object-states** without catastrophic forgetting. Encourages modular composition of learned representations for continual learning.
- **Keywords:** continual learning, compositional generalization, object-state learning, memory retention    

---

## **2023**

### [Differentiable Tree Operations Promote Compositional Generalization](https://proceedings.mlr.press/v202/soulos23a/soulos23a.pdf)

- **Year:** 2023
- **Conference/Venue:** ICML 2023 / PMLR Proceedings
- **Institutions:** (Unspecified)
- **Abstract:** Introduces **differentiable tree operations** using tensor-product representations to incorporate symbolic tree structure within neural networks. Enables end-to-end learning of compositional tree transformations with better generalization.
- **Keywords:** neural-symbolic models, tree structure, compositionality, differentiable operations, generalization    

---

## **2022**

### [Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale](https://aclanthology.org/2022.tacl-1.81.pdf)

- **Year:** 2022
- **Conference/Venue:** _Transactions of the ACL (TACL)_
- **Institutions:** (Unspecified)
- **Abstract:** Proposes **Transformer Grammars**, integrating grammar-based inductive biases into large-scale Transformer LMs to improve syntactic generalization and structural awareness.
- **Keywords:** transformer, syntax, inductive bias, grammar augmentation, structured generalization    

---

## **2021**

### [The Devil is in the Details: Simple Tricks Improve Systematic Generalization of Transformers](https://aclanthology.org/2021.emnlp-main.49.pdf)

- **Year:** 2021
- **Conference/Venue:** EMNLP 2021
- **Institutions:** (Unspecified)
- **Abstract:** Demonstrates that **simple implementation changes** (e.g., normalization, positional encoding, training curriculum) substantially enhance systematic generalization in Transformers, without architectural changes.
- **Keywords:** systematic generalization, transformer, implementation details, compositionality    

### [Compositional Generalization for Neural Semantic Parsing via Span-level Supervision](https://aclanthology.org/2021.naacl-main.225.pdf)

- **Year:** 2021    
- **Conference/Venue:** NAACL 2021
- **Institutions:** (Unspecified)
- **Abstract:** Improves compositional generalization in neural semantic parsing via **span-level supervision**, guiding intermediate representation learning for unseen logical compositions.
- **Keywords:** semantic parsing, compositional generalization, span supervision, sequence-to-sequence models

---

## **2017**

### [Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks](https://arxiv.org/pdf/1711.00350)

- **Year:** 2017
- **Conference/Venue:** arXiv preprint
- **Institutions:** (Unspecified)
- **Abstract:** A foundational study showing that standard **seq2seq RNNs** fail to achieve systematic compositional generalization, despite good interpolation performance — motivating later work on structure-aware and modular architectures.
- **Keywords:** compositionality, RNN, systematicity, sequence-to-sequence learning, generalization