# Papers

## 2025

### [Algorithmic Primitives and Compositional Geometry of Reasoning in Language Models](local:///mnt/data/2510.15987v1.pdf)

- **Year:** 2025    
- **Conference/Venue:** Preprint (arXiv 2025)
- **Institutions:** Columbia University; UCLA; Microsoft Research; multiple universities
- **Abstract:** Presents a framework for identifying **algorithmic primitives** underlying multi-step reasoning. Uses clustering of internal activations and function-vector methods to extract **primitive vectors** for tasks like TSP, 3SAT, AIME, and graph navigation. Shows these primitives support **geometric composition** (addition, subtraction, scaling) and transfer across tasks and models, with reasoning-finetuned models displaying stronger compositional generalization.    
- **Keywords:** algorithmic primitives, function vectors, reasoning geometry, compositionality, mechanistic interpretability.

### [Task Vectors in In-Context Learning: Emergence, Formation, and Benefits](local:///mnt/data/2501.09240v1.pdf)

- **Year:** 2025    
- **Conference/Venue:** Preprint (arXiv 2025)
- **Institutions:** University of Wisconsin–Madison; Microsoft Research
- **Abstract:** Studies **task-vector emergence** in transformers trained from scratch on synthetic tasks. Shows task vectors naturally appear but may be weak or entangled with query information. Introduces **TVP-loss**, a training method that strengthens and localizes task-specific vectors, improving zero-shot task-vector prompting and robustness without harming ICL performance.    
- **Keywords:** task vectors, synthetic training, TVP-loss, task-vector prompting, robustness, mechanistic analysis.

---

## 2024
### [Do Large Language Models Have Compositional Ability? An Investigation into Limitations and Scalability](https://arxiv.org/pdf/2407.15720)

- **Year:** 2024  
- **Conference/Venue:** Conference on Large-scale Models (COLM 2024)  
- **Institutions:** University of Wisconsin–Madison  
- **Abstract:** The authors build a suite of composite tasks (linguistic and logical) to study LLMs’ in-context learning on compositions of simple tasks. They observe decent performance (improving with scale) on simpler, separable compositions, but underperformance on multi-step reasoning compositions where scaling gives little benefit; they provide theory explaining when separability enables compositional capability.  
- **Keywords:** compositional ability, in-context learning, composite tasks, separable tasks, scaling laws, theoretical analysis, LLM evaluation.  

### [Skills-in-Context: Unlocking Compositionality in Large Language Models](https://aclanthology.org/2024.findings-emnlp.812.pdf)

- **Year:** 2024  
- **Conference/Venue:** Findings of the Association for Computational Linguistics: EMNLP 2024  
- **Institutions:** Tencent AI Lab (Bellevue, WA, USA); (author affiliation note: one author affiliated with Georgia Institute of Technology)  
- **Abstract:** Proposes **skills-in-context (SKiC)** prompts that demonstrate foundational skills and composed examples within one context. With as few as two exemplars, SKiC yields near-perfect systematic generalization across tasks, transfers well, and—when used for fine-tuning—enables zero-shot weak-to-strong generalization.  
- **Keywords:** skills-in-context, in-context learning, compositional generalization, systematic generalization, prompting, fine-tuning, weak-to-strong generalization.  

### [Function Vectors in Large Language Models](local:///mnt/data/2310.15213v2.pdf)

- **Year:** 2024
- **Conference/Venue:** ICLR 2024
- **Institutions:** Northeastern University
- **Abstract:** Demonstrates that transformer LMs internally form **function vectors (FVs)**—compact activation-space representations of tasks—transported by a small set of causal attention heads during in-context learning. FVs generalize across contexts, enabling zero-shot task execution and supporting limited **vector arithmetic** to combine tasks.
- **Keywords:** function vectors, in-context learning, causal mediation, task induction, mechanistic interpretability.

---

## 2023
### [Compositional Exemplars for In-context Learning](https://arxiv.org/pdf/2302.05698)

- **Year:** 2023  
- **Conference/Venue:** Proceedings of the 40th International Conference on Machine Learning (ICML 2023)  
- **Institutions:** The University of Hong Kong; Shanghai Artificial Intelligence Laboratory  
- **Abstract:** Introduces **CEIL (Compositional Exemplars for In-context Learning)**, a determinantal point process–based approach that models interactions among in-context examples rather than selecting them independently. By optimizing a contrastive objective with LM feedback, CEIL selects diverse yet relevant examples that improve generalization across 12 NLP datasets spanning classification, commonsense reasoning, code generation, and semantic parsing. Results show state-of-the-art performance and compositional transferability across datasets and LMs.  
- **Keywords:** in-context learning, exemplar selection, compositional generalization, determinantal point processes, diversity, transferability, retrieval learning.  

### [How Do In-Context Examples Affect Compositional Generalization?](https://aclanthology.org/2023.acl-long.618.pdf)

- **Year:** 2023  
- **Conference/Venue:** Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), Long Papers  
- **Institutions:** Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University; Microsoft Corporation  
- **Abstract:** The paper presents **COFE**, a test suite for in-context compositional generalization, and finds performance is highly sensitive to the chosen in-context examples. Effective examples are structurally similar to the test case, diverse from each other, and individually simple; challenges remain for fictional words and coverage of required linguistic structures.  
- **Keywords:** in-context learning, compositional generalization, COFE, example selection, similarity, diversity, complexity, semantic parsing.  

### [In-Context Learning Creates Task Vectors](local:///mnt/data/2310.15916v1.pdf)

- **Year:** 2023
- **Conference/Venue:** Preprint (arXiv 2023)
- **Institutions:** Tel Aviv University; Google DeepMind    
- **Abstract:** Provides a hypothesis-class interpretation of ICL by showing that LLMs compress demonstrations into a **task vector θ(S)**, which is independent of the query and modulates later layers to implement the learned rule. Experiments verify this factorization across many tasks, revealing that ICL resembles learning a parameterized function in activation space.
- **Keywords:** task vectors, hypothesis class, ICL mechanism, activation patching, mechanistic decomposition.