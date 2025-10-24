# Papers

## 2025

### [New Evidence of the Two-Phase Learning Dynamics of Neural Networks](https://arxiv.org/pdf/2505.13900)

* **Year:** 2025
* **Conference/Venue:** International Conference on Learning Representations (**ICLR 2025**, Workshop Track — DeLTa)
* **Institutions:** Shanghai Jiao Tong University; University of Michigan; National Institute of Informatics; SOKENDAI
* **Abstract:** Provides structural evidence for the **two-phase learning dynamics** observed in deep networks. Through interval-wise analysis of parameter evolution, the authors identify two phenomena: the *Chaos Effect*—a sensitive early stage where perturbations cause divergent learning trajectories—and the *Cone Effect*—a later phase where model updates become constrained within a functional subspace. The framework explains the transition from exploration to convergence in neural training.
* **Keywords:** training dynamics, phase transitions, chaos effect, cone effect, stability, interpretability, learning process.

### [Swing-by Dynamics in Concept Learning and Compositional Generalization](https://arxiv.org/pdf/2410.08309)
- **Year:** 2025  
- **Conference/Venue:** International Conference on Learning Representations (**ICLR 2025**)  
- **Institutions:** Harvard University (CBS-NTT Physics of Intelligence Program), NTT Research, University of Michigan  
- **Authors:** Yongyi Yang, Core Francisco Park, Ekdeep Singh Lubana, Maya Okawa, Wei Hu, Hidenori Tanaka  
- **Abstract:**  
  This paper introduces a theoretical and empirical framework to explain **compositional generalization** and emergent structure learning in generative models. Building on empirical findings from text-conditioned diffusion models, the authors define a simplified abstraction—the **Structured Identity Mapping (SIM)** task—where a network learns identity mappings over a Gaussian mixture with structured centroids representing “concepts.” The SIM task captures key empirical phenomena observed in multimodal models, including sequential learning of concepts and sensitivity to data structure.  

  The paper discovers **Swing-by Dynamics**, a new mechanism underlying **non-monotonic learning curves** and multiple-descent behaviors in out-of-distribution (OOD) generalization. Analytical results on deep linear models (MLPs and symmetric 2-layer networks) reveal alternating stages of growth and suppression in the model Jacobian, corresponding to transient memorization and recovery phases—mirroring behavior in diffusion models.  

  Empirical experiments with diffusion models confirm these theoretical predictions: models exhibit sequential concept learning, double-descent-like OOD loss curves, and exponential slowing in learning speed. Together, these results connect the **geometry of compositional data**, **training dynamics**, and **emergent generalization** in modern generative systems.  
- **Keywords:** compositional generalization, concept learning, diffusion models, swing-by dynamics, multiple descent, learning dynamics, deep linear networks, representation geometry.

## 2024

### [Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space](https://proceedings.neurips.cc/paper_files/paper/2024/file/99e6bcf460ea36818cf236da29311e73-Paper-Conference.pdf)

* **Year:** 2024
* **Conference/Venue:** 38th Conference on Neural Information Processing Systems (**NeurIPS 2024**)
* **Institutions:** Harvard University; NTT Research; University of Michigan
* **Abstract:** Introduces a **concept-space framework** for analyzing the internal learning geometry of neural networks. Capabilities emerge when the model’s concept coordinates realign, showing abrupt shifts corresponding to the onset of new implicit skills. The paper formalizes **concept signal** as a metric for concept-level learning speed and identifies hidden capability emergence points across diverse model architectures.
* **Keywords:** concept space, hidden capabilities, learning dynamics, interpretability, emergence, neural geometry.

### [Learning to Grok: Emergence of In-Context Learning and Skill Composition in Modular Arithmetic Tasks](https://proceedings.neurips.cc/paper_files/paper/2024/file/17d60fef592086d1a5cb136f1946df59-Paper-Conference.pdf)

* **Year:** 2024
* **Conference/Venue:** 38th Conference on Neural Information Processing Systems (**NeurIPS 2024**)
* **Institutions:** University of Maryland; Meta AI
* **Abstract:** Analyzes how transformers acquire **in-context learning (ICL)** through a grokking-like process in modular arithmetic tasks. Reveals four generalization phases—memorization, interpolation, representation reorganization, and abstraction—and identifies **circular modular embeddings** and a **representation phase transition**. Demonstrates that scale and task variety accelerate grokking and composition.
* **Keywords:** grokking, in-context learning, modular arithmetic, emergent reasoning, phase transition, representational geometry.

### [Grokking as the Transition from Lazy to Rich Training Dynamics](https://arxiv.org/pdf/2310.06110)
- **Year:** 2024  
- **Conference/Venue:** International Conference on Learning Representations (**ICLR 2024**)  
- **Institutions:** Harvard University; Kempner Institute for the Study of Natural and Artificial Intelligence  
- **Authors:** Tanishq Kumar, Blake Bordelon, Samuel J. Gershman*, Cengiz Pehlevan* (*Equal Senior Authors)  
- **Abstract:**  
  This paper proposes that **grokking**—where a neural network’s training loss falls long before its test loss—arises from a transition between **lazy training** and **rich (feature-learning) regimes**. Through theoretical analysis and controlled experiments on polynomial regression and modular arithmetic tasks, the authors show that grokking occurs when networks initially operate as **linearized models** (with fixed features) before eventually learning new representations that generalize.  
  They identify two key control parameters:  
  1. **Network laziness (α):** a scaling factor governing the rate of feature learning.  
  2. **NTK–task alignment:** the overlap between the initial neural tangent kernel and the target function, quantifiable via **centered kernel alignment (CKA)**.  
  Grokking is most pronounced when models begin in the lazy regime with low kernel–task alignment and moderate dataset sizes. The framework generalizes across architectures—including MLPs, Transformers, and student–teacher setups—without relying on weight decay.  
- **Keywords:** grokking, lazy training, feature learning, neural tangent kernel (NTK), centered kernel alignment (CKA), generalization dynamics, polynomial regression, scaling laws.

### [Grokking as a First-Order Phase Transition in Two-Layer Networks](https://arxiv.org/pdf/2310.03789)
- **Year:** 2024  
- **Conference/Venue:** International Conference on Learning Representations (**ICLR 2024**)  
- **Institutions:** Racah Institute of Physics, Hebrew University of Jerusalem; Tel Aviv University  
- **Abstract:** Provides a theoretical framework showing that **grokking**—the delayed generalization phenomenon where networks suddenly acquire test accuracy after long plateaus—is a **first-order phase transition** in the feature space of neural networks. Using mean-field theory and adaptive kernel analysis, the authors demonstrate that training dynamics undergo a sharp **representation phase transition** from random Gaussian features to feature-aligned mixtures as networks begin to generalize. This transition exhibits the hallmarks of a thermodynamic phase change, with coexistence of memorization and generalization phases and abrupt realignment of internal representations.  
- **Keywords:** grokking, phase transition, feature learning, adaptive kernel theory, representation dynamics, generalization, statistical mechanics.

## 2023

### [Progress Measures for Grokking via Mechanistic Interpretability](https://openreview.net/pdf?id=9XFSbDPmdW)
* **Year:** 2023  
* **Conference/Venue:** International Conference on Learning Representations (**ICLR 2023**)  
* **Institutions:** Independent researchers; University of California, Berkeley  
* **Abstract:** Reverse-engineers small transformers trained on modular addition and shows they implement a **Fourier-based algorithm** (mapping tokens to sinusoidal features and combining them via trig identities). From this mechanism, the authors define two **progress measures**—**restricted loss** (keep only key Fourier components) and **excluded loss** (remove only key components)—that evolve smoothly before the apparent “snap” in test accuracy. Training decomposes into three phases: **memorization**, **circuit formation**, and **cleanup**, with **weight decay** driving the final shift from memorization to a sparse, generalizing circuit—explaining grokking as a gradual mechanism rather than a discontinuity. :contentReference[oaicite:0]{index=0}  
* **Keywords:** grokking, mechanistic interpretability, Fourier features, progress measures, modular addition, transformers, weight decay, phase transitions.
