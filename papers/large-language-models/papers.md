# Papers

## 2025

### [Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](https://aclanthology.org/2025.acl-long.1508.pdf)
- **Year:** 2025  
- **Conference/Venue:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Long Papers  
- **Institutions:** Nara Institute of Science and Technology (NAIST), Japan  
- **Abstract:** Introduces **Ordered CommonGen**, a benchmark assessing both instruction following and compositionality by requiring concept order adherence. Across 36 LLMs, models often bias toward particular order patterns, reducing diversity; even the best model attains ~75% ordered coverage, indicating room to improve both abilities.  
- **Keywords:** instruction following, compositional generalization, Ordered CommonGen, concept order, ordered coverage, commonsense reasoning, LLM benchmarking.  

### [Evaluating Morphological Compositional Generalization in Large Language Models](https://aclanthology.org/2025.naacl-long.59.pdf)
- **Year:** 2025  
- **Conference/Venue:** Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2025), Volume 1: Long Papers  
- **Institutions:** EPFL; Idiap Research Institute; Duke University; Brandeis University; LMU Munich; University of Cambridge; Johns Hopkins University; Università della Svizzera Italiana; New York University :contentReference[oaicite:0]{index=0}  
- **Abstract:** Defines morphemes as **compositional primitives** and introduces generative (productivity) and discriminative (systematicity) tasks to probe morphological generalization. Focusing on **Turkish** and **Finnish**, the study evaluates instruction-tuned multilingual LLMs (e.g., GPT-4, Gemini, Aya, Qwen). Models struggle especially on **novel roots** and as **morphological complexity** increases: productivity accuracy drops toward zero with longer affix chains, and systematicity shows large gaps in a stringent **coherence** metric compared to humans, despite above-chance performance on individual combinations. Overall, LLMs lag far behind human **morphological compositionality** in agglutinative languages. :contentReference[oaicite:1]{index=1}  
- **Keywords:** morphology, compositional generalization, productivity, systematicity, agglutinative languages, Turkish, Finnish, LLM evaluation, coherence metric, nonce/Wug testing.

---

## 2024

### [Rule Extrapolation in Language Models: A Study of Compositional Generalization on OOD Prompts](https://arxiv.org/pdf/2409.13728)
- **Year:** 2024  
- **Conference/Venue:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024)  
- **Institutions:** University of Cambridge; Max Planck Institute for Intelligent Systems (Tübingen); ELLIS Institute Tübingen; Tübingen AI Center; AI Center, UCL  
- **Abstract:** Defines **rule extrapolation**—OOD scenarios for formal languages where prompts violate one rule—and evaluates linear/recurrent models, Transformers, and state-space models. No single architecture dominates: Transformers excel on context-free/sensitive languages while others do better on regular ones; a normative perspective draws on the Solomonoff prior from algorithmic information theory.  
- **Keywords:** out-of-distribution generalization, rule extrapolation, formal languages, Transformers, state space models, simplicity bias, Solomonoff prior.  

---

## 2023

### [Compositional Task Representations for Large Language Models](https://openreview.net/pdf?id=6axIMJA7ME3)
- **Year:** 2023  
- **Conference/Venue:** International Conference on Learning Representations (ICLR 2023)  
- **Institutions:** Recurrent AI; Beijing Jiaotong University; Tsinghua University; Shanghai Artificial Intelligence Laboratory; Shanghai Qi Zhi Institute  
- **Abstract:** Large language models have shown a remarkable cross-task generalization ability. Most prior works assumed that prompts extract knowledge from language models to facilitate generalization to new tasks. In contrast, the paper introduces **compositional task representations (CTR)**—a prompt-free approach that learns a discrete, compositional codebook via multi-task training—and shows CTR substantially outperforms prompt-based methods in zero-label learning; several codes are interpretable and controllable.  
- **Keywords:** compositional generalization, task codebook, multi-task learning, prompt-free, zero-label learning, controllability, large language models.  