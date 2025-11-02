## 2025

### [Understanding Subword Compositionality of Large Language Models](https://arxiv.org/pdf/2508.17953v1)
- **Year:** 2025  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** University of Copenhagen, ETH Zurich  
- **Abstract:** Analyzes how LLMs construct meaningful word representations from subword components. Through geometry and probing analyses across six LLMs, the study identifies three distinct subword composition strategies affecting structural similarity, semantic decomposability, and form retention. Findings show that most models form nearly additive (isometric) compositions and preserve semantic information but vary in how they retain surface features like word length.  
- **Keywords:** subword compositionality, structural similarity, semantic decomposability, morphology, word representation, probing analysis, LLM embeddings.  

---

### [CHARBENCH: Evaluating the Role of Tokenization in Character-Level Tasks](https://arxiv.org/pdf/2508.02591v2)
- **Year:** 2025  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** Ben-Gurion University of the Negev, Israel  
- **Abstract:** Introduces **CHARBENCH**, a large-scale benchmark for evaluating character-level reasoning in LLMs. Comprising counting and indexing tasks over two orders of magnitude larger than previous datasets, CHARBENCH reveals that LLMs perform poorly (avg. 43.6% accuracy), especially on positional reasoning. Analysis shows that token length correlates negatively with accuracy in intra-word tasks, suggesting subword tokenization obscures fine-grained character information.  
- **Keywords:** character-level reasoning, tokenization analysis, benchmark, subword segmentation, intra-word representation, linguistic evaluation, LLM performance.  

---

### [ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space](https://arxiv.org/pdf/2508.02247v2)
- **Year:** 2025  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** Stevens Institute of Technology, USA  
- **Abstract:** Proposes **ByteGen**, a tokenizer-free generative model for high-frequency limit order book (LOB) event streams. ByteGen models market dynamics directly in **byte space** using a hybrid Mamba-Transformer (H-Net) architecture, eliminating the need for discretization or tokenization. Trained on 34M CME Bitcoin futures events, ByteGen reproduces stylized market facts while preserving full numerical precision and long-term dependencies.  
- **Keywords:** tokenizer-free modeling, byte-level generation, orderbook simulation, financial data modeling, H-Net, Mamba-Transformer, high-frequency trading.  

---

### [Generative Recommender with End-to-End Learnable Item Tokenization](https://arxiv.org/pdf/2409.05546v3)
- **Year:** 2025  
- **Conference/Venue:** Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)  
- **Institutions:** Renmin University of China, Kuaishou Technology  
- **Abstract:** Proposes **ETEGRec**, an end-to-end generative recommender that unifies item tokenization and autoregressive recommendation training. Unlike previous two-stage methods, ETEGRec jointly optimizes both components through **sequence-item** and **preference-semantic alignment**, enabling mutual enhancement between tokenization and generation. Experiments demonstrate significant gains over traditional and generative baselines across multiple datasets.  
- **Keywords:** generative recommendation, end-to-end training, item tokenization, sequence modeling, preference alignment, RQ-VAE, T5 architecture.  

---

### [UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/pdf/2502.20321v3)
- **Year:** 2025  
- **Conference/Venue:** Advances in Neural Information Processing Systems (NeurIPS 2025)  
- **Institutions:** The University of Hong Kong, ByteDance Inc., Huazhong University of Science and Technology  
- **Abstract:** Introduces **UniTok**, a unified visual tokenizer bridging image generation and understanding. By integrating reconstruction (VQVAE) and semantic (CLIP) supervision via **multi-codebook quantization**, UniTok overcomes representational bottlenecks in discrete token spaces. It achieves state-of-the-art reconstruction and zero-shot performance while enabling multimodal LLMs with native visual generation capabilities.  
- **Keywords:** unified visual tokenizer, VQVAE, CLIP, multi-codebook quantization, multimodal LLM, visual generation, semantic supervision.  

---

### [Towards Semantic Equivalence of Tokenization in Multimodal LLM](https://arxiv.org/pdf/2406.05127v4)
- **Year:** 2025  
- **Conference/Venue:** International Conference on Learning Representations (ICLR 2025)  
- **Institutions:** National University of Singapore, ByteDance Seed, Nanyang Technological University, Skywork AI  
- **Abstract:** Presents **SeTok**, a **Semantic-Equivalent Vision Tokenizer** for multimodal LLMs. SeTok dynamically clusters visual embeddings into semantically meaningful units, enabling closer alignment between visual and linguistic tokens. Integrated into **SETOKIM**, this approach improves visual understanding, segmentation, and generation through fine-grained semantic alignment.  
- **Keywords:** multimodal tokenization, semantic equivalence, dynamic clustering, SeTok, MLLM, visual semantics, image-text alignment.  

## 2024

### [An Analysis of Tokenization: Transformers under Markov Data](https://proceedings.neurips.cc/paper_files/paper/2024/file/724afcaae4ae92a9220a077ffe80088d-Paper-Conference.pdf)
- **Year:** 2024  
- **Conference/Venue:** Advances in Neural Information Processing Systems (NeurIPS 2024)  
- **Institutions:** University of California, Berkeley  
- **Abstract:** Presents an empirical and theoretical analysis of how tokenization impacts transformers trained on data from Markov sources. Shows that, without tokenization, transformers behave as unigram models with high cross-entropy, while tokenized models can approximate optimal distributions. The work formally connects dictionary size, cross-entropy loss, and Markov order, providing the first theoretical framework for why tokenization improves learning efficiency.  
- **Keywords:** tokenization theory, Markov modeling, transformers, language modeling, cross-entropy, BPE, statistical analysis.  

---

### [Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs](https://arxiv.org/pdf/2402.14903v1)
- **Year:** 2024  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** Gatsby Computational Neuroscience Unit (UCL), Google DeepMind  
- **Abstract:** Examines how number tokenization schemes (e.g., single-digit vs multi-digit) influence arithmetic reasoning in models like GPT-3.5 and GPT-4. The study finds that **right-to-left (comma-enforced)** tokenization improves accuracy up to 99%, suggesting systematic computation patterns rather than approximation. Highlights tokenization-induced inductive biases and scaling effects.  
- **Keywords:** number tokenization, numerical reasoning, GPT-3.5, GPT-4, inductive bias, arithmetic evaluation, token directionality.  

---

### [Evaluating Subword Tokenization: Alien Subword Composition and OOV Generalization Challenge](https://arxiv.org/pdf/2404.13292v1)
- **Year:** 2024  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** University of Melbourne, National University of Mongolia, University of Edinburgh, Ben-Gurion University of the Negev, IMT Atlantique  
- **Abstract:** Introduces **umLabeller**, a 98%-accurate tool for classifying subword compositions as morphological or “alien.” Paired with the **OOV Generalization Challenge**, it evaluates how subword tokenization affects semantic compositionality and generalization. Results show that alien tokenization degrades performance in downstream NLP tasks.  
- **Keywords:** subword tokenization, morphological segmentation, OOV generalization, umLabeller, BPE, Unigram LM, alien composition, semantic compositionality.  

---

### [Tokenization Falling Short: On Subword Robustness in Large Language Models](https://aclanthology.org/2024.findings-emnlp.86.pdf)
- **Year:** 2024  
- **Conference/Venue:** Findings of the Association for Computational Linguistics: EMNLP 2024  
- **Institutions:** Baidu, ModelBest, University of Copenhagen  
- **Abstract:** Investigates the *“curse of tokenization”* — LLM sensitivity to typos, token length, and internal token structure ignorance. The study evaluates LLMs across tasks such as anagram solving, mathematical reasoning, and typographical robustness, revealing persistent weaknesses despite model scaling. Regularized subword tokenization (e.g., BPE-dropout) improves robustness but does not eliminate these issues.  
- **Keywords:** tokenization robustness, subword encoding, typographical variation, compositional structure, BPE-dropout, LLM evaluation, linguistic perturbation.  

---

### [NumeroLogic: Number Encoding for Enhanced LLMs’ Numerical Reasoning](https://aclanthology.org/2024.emnlp-main.12.pdf)
- **Year:** 2024  
- **Conference/Venue:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)  
- **Institutions:** IBM Research, MIT-IBM Watson AI Lab, MIT  
- **Abstract:** Introduces **NumeroLogic**, a preprocessing-based tokenization reform that prefixes numbers with their digit counts (e.g., “42” → “2:42”), allowing models to infer digit place values during generation. This format acts as a lightweight *Chain of Thought*, improving arithmetic reasoning and general performance in language modeling benchmarks like MMLU, without altering model architectures.  
- **Keywords:** numerical reasoning, number tokenization, digit encoding, causal language models, Chain of Thought, BPE, LoRA fine-tuning.  

---

### [Assessing the Importance of Frequency versus Compositionality for Subword-based Tokenization in NMT](https://arxiv.org/pdf/2306.01393v3)
- **Year:** 2024  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** HEIG-VD / HES-SO, EPFL, Armasuisse W+T (Switzerland)  
- **Abstract:** Disentangles the effects of **frequency** and **compositionality** in subword tokenization using a Huffman coding–based tokenizer. Experiments in Neural Machine Translation (NMT) show frequency alone explains 90–95% of BPE’s success, suggesting compositionality plays a minor role.  
- **Keywords:** subword tokenization, Byte-Pair Encoding (BPE), Huffman coding, frequency effects, compositionality, neural machine translation (NMT).  

---

### [MorphPiece: A Linguistic Tokenizer for Large Language Models](https://arxiv.org/pdf/2307.07262v2)
- **Year:** 2024  
- **Conference/Venue:** arXiv preprint  
- **Institutions:** Ludwig Maximilian University of Munich, Germany  
- **Abstract:** Proposes **MorphPiece**, a linguistically grounded tokenizer combining morphological segmentation with BPE. The resulting **MorphGPT** model surpasses GPT-2 across several NLP benchmarks (GLUE, MTEB), demonstrating the benefits of morphology-aware segmentation over purely statistical subword tokenization.  
- **Keywords:** morphological tokenization, MorphPiece, MorphGPT, linguistic segmentation, subword modeling, GLUE benchmark, MTEB, morphological analysis.  

## 2023

### [Inducing Character-level Structure in Subword-based Language Models with Type-level Interchange Intervention Training](https://aclanthology.org/2023.findings-acl.770.pdf)
- **Year:** 2023  
- **Conference/Venue:** Findings of the Association for Computational Linguistics: ACL 2023  
- **Institutions:** Stanford University, The University of Texas at Austin  
- **Abstract:** Proposes a **causal intervention framework** (Type-level Interchange Intervention Training) that teaches subword-based LMs to internalize character-level representations, improving performance on character manipulation tasks (e.g., spelling correction, word games). The approach enhances robustness on unseen token sequences and yields interpretable internal character representations.  
- **Keywords:** subword tokenization, character-level modeling, causal abstraction, interchange intervention training (IIT), spelling correction, compositionality. 
