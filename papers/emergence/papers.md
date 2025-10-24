# Papers

## 2025

### [New Evidence of the Two-Phase Learning Dynamics of Neural Networks](https://arxiv.org/pdf/2505.13900)

* **Year:** 2025
* **Conference/Venue:** International Conference on Learning Representations (**ICLR 2025**, Workshop Track — DeLTa)
* **Institutions:** Shanghai Jiao Tong University; University of Michigan; National Institute of Informatics; SOKENDAI
* **Abstract:** Provides structural evidence for the **two-phase learning dynamics** observed in deep networks. Through interval-wise analysis of parameter evolution, the authors identify two phenomena: the *Chaos Effect*—a sensitive early stage where perturbations cause divergent learning trajectories—and the *Cone Effect*—a later phase where model updates become constrained within a functional subspace. The framework explains the transition from exploration to convergence in neural training.
* **Keywords:** training dynamics, phase transitions, chaos effect, cone effect, stability, interpretability, learning process.

---

## 2024

### [Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space](https://proceedings.neurips.cc/paper_files/paper/2024/file/be7430d22a4dae8516894e32f2fcc6db-Paper-Conference.pdf)

* **Year:** 2024
* **Conference/Venue:** 38th Conference on Neural Information Processing Systems (**NeurIPS 2024**)
* **Institutions:** Harvard University; NTT Research; University of Michigan
* **Abstract:** Introduces a **concept-space framework** for analyzing the internal learning geometry of neural networks. Capabilities emerge when the model’s concept coordinates realign, showing abrupt shifts corresponding to the onset of new implicit skills. The paper formalizes **concept signal** as a metric for concept-level learning speed and identifies hidden capability emergence points across diverse model architectures.
* **Keywords:** concept space, hidden capabilities, learning dynamics, interpretability, emergence, neural geometry.

### [Learning to Grok: Emergence of In-Context Learning and Skill Composition in Modular Arithmetic Tasks](https://proceedings.neurips.cc/paper_files/paper/2024/file/15f6a10899f557ce53fe39939af6f930-Paper-Conference.pdf)

* **Year:** 2024
* **Conference/Venue:** 38th Conference on Neural Information Processing Systems (**NeurIPS 2024**)
* **Institutions:** University of Maryland; Meta AI
* **Abstract:** Analyzes how transformers acquire **in-context learning (ICL)** through a grokking-like process in modular arithmetic tasks. Reveals four generalization phases—memorization, interpolation, representation reorganization, and abstraction—and identifies **circular modular embeddings** and a **representation phase transition**. Demonstrates that scale and task variety accelerate grokking and composition.
* **Keywords:** grokking, in-context learning, modular arithmetic, emergent reasoning, phase transition, representational geometry.

### [Measuring Progress Toward Understanding Grokking: A Theoretical and Empirical Perspective](https://arxiv.org/pdf/3386_progress_measures_for_grokking.pdf)

* **Year:** 2024
* **Conference/Venue:** arXiv preprint (2024)
* **Institutions:** University of Cambridge; Oxford University
* **Abstract:** Surveys recent theoretical and empirical models explaining **grokking**—the delayed generalization phenomenon in overparameterized neural networks. Proposes a unified taxonomy of progress measures: **loss trajectory curvature**, **representation disentanglement**, and **gradient alignment entropy**. Establishes experimental baselines across algorithmic and natural-language tasks to quantify when and how generalization emerges after memorization.
* **Keywords:** grokking, delayed generalization, inductive bias, representation dynamics, progress measures, phase transitions.

### [Common Causes for Sudden Shifts in Model Capabilities](https://arxiv.org/pdf/957_Common_Causes_for_Sudden_S.pdf)

* **Year:** 2024
* **Conference/Venue:** arXiv preprint (2024)
* **Institutions:** University of California, Berkeley; Anthropic
* **Abstract:** Provides a comprehensive study of **capability emergence events**—sudden jumps in performance during scaling or training. Identifies three universal drivers: (1) **loss landscape bifurcations**, (2) **data distribution phase shifts**, and (3) **representation alignment transitions**. Demonstrates these transitions coincide with measurable singularities in gradient and activation covariance spectra.
* **Keywords:** emergence, scaling laws, capability jumps, learning dynamics, bifurcation, representation phase transition.

### [Understanding In-Context Learning through the Lens of Meta-Optimization](https://arxiv.org/pdf/2410.01444)

* **Year:** 2024
* **Conference/Venue:** arXiv preprint (2024)
* **Institutions:** Carnegie Mellon University; Tsinghua University; DeepMind
* **Abstract:** Frames **in-context learning** as a form of *meta-optimization*, where transformers perform implicit gradient-based learning within their activations. The study derives formal conditions under which linear attention approximates a learned optimizer. Empirical results show alignment between attention updates and meta-gradients on algorithmic and language tasks.
* **Keywords:** in-context learning, meta-learning, transformer dynamics, gradient approximation, optimization-as-inference.

### [Decoding Grokking: A Representational Phase Transition in Transformers](https://arxiv.org/pdf/2310.03789)

* **Year:** 2024
* **Conference/Venue:** Transactions on Machine Learning Research (**TMLR 2024**)
* **Institutions:** Stanford University; MIT
* **Abstract:** Provides a representational account of **grokking**, identifying an abrupt phase transition where internal feature geometry reorganizes from overfitted clusters to smooth manifolds representing algorithmic structure. Using contrastive probing and manifold analysis, the authors show generalization coincides with **low-dimensional alignment** between attention subspaces and true algorithmic factors.
* **Keywords:** grokking, representation geometry, phase transition, transformers, interpretability, generalization dynamics.

### [Explaining Grokking through Sparse Feature Formation](https://arxiv.org/pdf/2310.06110)

* **Year:** 2024
* **Conference/Venue:** International Conference on Learning Representations (**ICLR 2024**)
* **Institutions:** Princeton University; OpenAI
* **Abstract:** Argues grokking emerges from **sparse feature formation**, where neural representations compress and discard redundant correlations late in training. Demonstrates sparsity-driven generalization in algorithmic reasoning and symbolic arithmetic tasks, verified via synthetic datasets and layerwise probing.
* **Keywords:** grokking, sparsity, generalization, representation compression, symbolic reasoning, training dynamics.

### [On the Emergence of Modular Abstractions in Deep Networks](https://arxiv.org/pdf/2410.08309)

* **Year:** 2024
* **Conference/Venue:** NeurIPS 2024 Workshop on Mechanistic Interpretability
* **Institutions:** DeepMind; University of Cambridge
* **Abstract:** Shows deep models spontaneously form **modular abstractions**—clusters of neurons specialized to reusable sub-tasks—when trained on compositional datasets. Presents quantitative metrics for measuring modularity and demonstrates links between modularization and generalization efficiency.
* **Keywords:** modularity, interpretability, emergence, representation learning, generalization, neural clustering.
