# Papers

## 2026

### [A Neurosymbolic Agent System for Compositional Visual Reasoning](https://openreview.net/pdf?id=ZSal26DrNb)

* **Year:** 2026
* **Conference/Venue:** Under review at the International Conference on Learning Representations (ICLR 2026)
* **Institutions:** Anonymous (under double-blind review)
* **Abstract:** Presents **VLAgent**, a two-stage neuro-symbolic system that plans (via LLM) and executes hybrid neural–symbolic programs for compositional visual reasoning. Introduces an **SS-parser** for syntax/semantic repair and an **execution verifier** for stepwise validation, showing strong gains on multiple visual reasoning benchmarks.
* **Keywords:** neuro-symbolic reasoning, visual question answering, compositional reasoning, LLM planning, interpretability, hybrid systems.

---

## 2025

### [Compositional Entailment Learning for Hyperbolic Vision-Language Models](https://arxiv.org/pdf/2410.06912.pdf)

* **Year:** 2025
* **Conference/Venue:** International Conference on Learning Representations (**ICLR 2025**)
* **Institutions:** University of Amsterdam; Sapienza University of Rome; ItalAI; Procederai
* **Abstract:** **HyCoCLIP** models hierarchical inter/intra-modal relations in hyperbolic space with contrastive + entailment-cone objectives, improving zero-shot and hierarchical generalization vs. CLIP/MERU.
* **Keywords:** hyperbolic embeddings, compositionality, entailment learning, vision-language models.

### [Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging](https://arxiv.org/pdf/2412.20070.pdf)

* **Year:** 2025
* **Conference/Venue:** arXiv preprint (2025)
* **Institutions:** The Chinese University of Hong Kong (Shenzhen)
* **Abstract:** Introduces **Med-MAT** (106 datasets) organized by **Modality–Anatomy–Task** triplets and shows multi-task training fosters compositional generalization in medical MLLMs.
* **Keywords:** multimodal LLMs, medical imaging, compositional generalization.

### [Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data](https://arxiv.org/pdf/2503.01167.pdf)

* **Year:** 2025
* **Conference/Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2025**)
* **Institutions:** Nanyang Technological University
* **Abstract:** **SPARCL** injects real-image features into fast T2I generation + style transfer and uses an adaptive margin loss, boosting CLIP compositionality by 5–8% across multiple benchmarks.
* **Keywords:** synthetic data, compositional learning, CLIP, adaptive margin loss.

### [Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models](https://arxiv.org/pdf/2503.17142.pdf)

* **Year:** 2025
* **Conference/Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2025**)
* **Institutions:** Fondazione Bruno Kessler; University of Trento; University of Twente
* **Abstract:** Proposes **Geodesically Decomposable Embeddings (GDE)** to capture manifold-structured visual compositionality, improving compositional classification and group robustness.
* **Keywords:** visual compositionality, Riemannian geometry, CLIP.

### [Unifying Symbolic and Neural Reasoning for Compositional Visual Tasks](https://arxiv.org/pdf/2504.21850.pdf)

* **Year:** 2025
* **Conference/Venue:** arXiv preprint (2025)
* **Institutions:** Tsinghua University; Beijing Academy of AI
* **Abstract:** A unified symbolic–neural framework with differentiable logic operations and structured attention yields interpretable, OOD-robust visual reasoning.
* **Keywords:** neuro-symbolic reasoning, logic programs, visual question answering.

### [Exploring the Role of Task Modularity in Multimodal Compositional Generalization](https://arxiv.org/pdf/2505.17955.pdf)

* **Year:** 2025
* **Conference/Venue:** arXiv preprint (2025)
* **Institutions:** Zhejiang University; Peking University; University of Cambridge
* **Abstract:** Analyzes how explicitly **modularizing tasks**—decomposing reasoning into functional sub-skills with structured prompting and modular fine-tuning—affects systematic generalization in multimodal settings. Across CLEVR/GQA-style evaluations, task modularity improves robustness and interpretability, especially for cross-task compositions.
* **Keywords:** modular architectures, compositional reasoning, multimodal models, systematic generalization, structured prompting.

### [Unveiling the Compositional Ability Gap in Vision-Language Reasoning Models](https://arxiv.org/pdf/2505.19406.pdf)

* **Year:** 2025
* **Conference/Venue:** arXiv preprint (2025)
* **Institutions:** CUHK; Tencent Hunyuan Research
* **Abstract:** Introduces **ComPABench**; finds RL post-training improves cross-task composition vs. SFT. **RL-Ground** (caption-before-thinking + grounding rewards) further narrows gaps.
* **Keywords:** RL for VLMs, compositional reasoning, grounding.

### [Multi-Sourced Compositional Generalization in Visual Question Answering](https://arxiv.org/pdf/2505.23045.pdf)

* **Year:** 2025
* **Conference/Venue:** International Joint Conference on Artificial Intelligence (**IJCAI 2025**)
* **Institutions:** Beijing Institute of Technology; Shenzhen MSU-BIT University
* **Abstract:** Defines **MSCG** across linguistic/visual primitives; proposes retrieval-augmented training and introduces **GQA-MSCG** with LL/VV/LV splits, improving cross-modal composition.
* **Keywords:** VQA, retrieval augmentation, multi-source composition.

### [Evaluating Compositional Generalisation in VLMs and Diffusion Models](https://arxiv.org/pdf/2508.20783.pdf)

* **Year:** 2025
* **Conference/Venue:** The Joint Conference on Lexical and Computational Semantics (***SEM 2025**)
* **Institutions:** University of Bristol; University of Amsterdam
* **Abstract:** Compares Diffusion Classifier, CLIP, ViLT on extended concept-binding tasks (ZSL/GZSL). Diffusion generalizes well in single-object settings; relational composition remains challenging for all.
* **Keywords:** diffusion models, VLMs, compositional evaluation, spatial relations.

---

## 2024

### [The Hard Positive Truth About Vision-Language Compositionality](https://arxiv.org/pdf/2409.17958)
- **Year:** 2024  
- **Conference/Venue:** European Conference on Computer Vision (**ECCV 2024**)  
- **Institutions:** University of Oxford; University of Edinburgh; University College London  
- **Abstract:** Revisits compositional evaluation in vision-language models (VLMs) and uncovers a systematic bias in widely used benchmarks caused by **hard positive pairs**—semantically correct but compositionally challenging samples. The authors demonstrate that models like CLIP and BLIP underperform on these cases even when easy negatives are well distinguished. They propose **Hard Positive Compositionality (HPC)**, a new evaluation protocol and curated dataset that isolates this difficulty. Results show HPC reveals performance drops of 15–30% masked by standard metrics, emphasizing the need for balanced positive sampling to assess true compositional understanding.  
- **Keywords:** compositional generalization, vision-language models, hard positives, evaluation bias, CLIP, BLIP, compositional benchmarking.

### [In-Context Compositional Generalization for Large Vision-Language Models](https://aclanthology.org/2024.emnlp-main.996.pdf)

* **Year:** 2024
* **Conference/Venue:** Empirical Methods in Natural Language Processing (**EMNLP 2024**)
* **Institutions:** Beijing Institute of Technology; Zhejiang University
* **Abstract:** Uses **diversity–coverage**-based demo selection to reduce redundancy and align multimodal context, improving ICCG on GQA-ICCG and VQA v2 across several LVLMs.
* **Keywords:** in-context learning, LVLMs, demonstration selection, ICCG.
