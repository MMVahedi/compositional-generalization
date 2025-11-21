# Papers

## 2025

### [Compositional Generalization via Forced Rendering of Disentangled Latents](https://arxiv.org/pdf/2501.18797)
- **Year:** 2025  
- **Conference/Venue:** ICML 2025 (PMLR 267)  
- **Abstract:** Shows that having disentangled latents alone doesn’t guarantee systematic generalization: standard decoders “re-entangle” later layers and rely on superposition/memorization. Forcing latents to render directly into the output representational (pixel) space—via architectural constraints, regularization, or curated data—yields data-efficient OOD composition on a controlled 2D “bump” task and analysis via kernels/manifold geometry.
- **Keywords:** compositional generalization, disentanglement, rendering constraint, memorization vs. composition, OOD.

### [Does Data Scaling Lead to Visual Compositional Generalization?](https://openreview.net/pdf?id=M2WMUuwoh5)
- **Year:** 2025  
- **Conference/Venue:** ICML 2025 (PMLR 267)  
- **Abstract:** In controlled vision setups, **diversity** (concept values and combination coverage) — not raw scale — drives compositional generalization. High combinatorial coverage induces **linearly factored** representations that enable perfect generalization from few combinations; pretrained DINO/CLIP show partial but imperfect structure.
- **Keywords:** data diversity, coverage, linearly factored reps, scaling laws, vision compositionality.

### [The Coverage Principle: A Framework for Understanding Compositional Generalization](https://arxiv.org/pdf/2505.20278)
- **Year:** 2025  
- **Conference/Venue:** arXiv preprint  
- **Abstract:** Proposes **coverage** as a necessary data-centric condition for pattern-matching models (e.g., Transformers) to generalize compositionally: reliable predictions extend only to inputs reachable by substituting **functionally equivalent** fragments observed in training. Predicts (and confirms) quadratic data growth for two-hop tasks, limits with path ambiguity, and partial gains from CoT; offers a taxonomy of mechanism types (structure-/property-/shared-operator-based).
- **Keywords:** coverage, functional equivalence, multi-hop scaling, path ambiguity, CoT limits.

### [When Does Compositional Structure Yield Compositional Generalization? A Kernel Theory.](https://openreview.net/pdf?id=FPBce2P1er)
- **Year:** 2025  
- **Conference/Venue:** ICLR 2025  
- **Abstract:** With **compositionally structured representations**, kernel models are provably constrained to **conjunction-wise additive** solutions (summing values assigned to seen component combinations). Identifies inherent limits (no transitive equivalence generalization) and failure modes (memorization leak, shortcut bias), then empirically shows deep nets exhibit matching behavior on similarly structured data.
- **Keywords:** kernel theory, compositional tasks, conjunction-wise additivity, dataset bias, theory-to-practice.

---

## 2024

### [Neuron Activation Coverage: Rethinking Out-of-Distribution Detection and Generalization](https://arxiv.org/pdf/2306.02879)
- **Year:** 2024  
- **Conference/Venue:** ICLR 2024  
- **Abstract:** Defines **Neuron Activation Coverage (NAC)**—a coverage measure over neuron states (combining output and decision influence)—and shows (i) strong OOD detection (SOTA across CIFAR-10/100, ImageNet) and (ii) positive correlation between NAC and generalization, enabling robust model selection beyond standard InD validation.
- **Keywords:** OOD detection, coverage metric, robustness evaluation, model selection.

### [Compositional Generalization Across Distributional Shifts with Sparse Tree Operations](local:///mnt/data/NeurIPS-2024-compositional-generalization-across-distributional-shifts-with-sparse-tree-operations-Paper-Conference.pdf)

- **Year:** 2024
- **Conference/Venue:** NeurIPS 2024
- **Abstract:** Introduces **Sparse Coordinate Trees (SCT)** and the **Sparse Differentiable Tree Machine (sDTM)**, a unified neurosymbolic architecture enabling differentiable tree operations in vector space. sDTM achieves robust compositional generalization across lexical, structural, and distribution-shift conditions while being dramatically more efficient (up to **75× fewer parameters**) than the original DTM. Demonstrates strong performance on Active↔Logical, FOR2LAM, SCAN, and GeoQuery.
- **Keywords:** neurosymbolic models, compositional generalization, SCT, sDTM, seq2seq, seq2tree, structural generalization.

---

## 2023

### [Improving Compositional Generalization using Iterated Learning and Simplicial Embeddings](https://proceedings.neurips.cc/paper_files/paper/2023/file/be7430d22a4dae8516894e32f2fcc6db-Paper-Conference.pdf)
- **Year:** 2023  
- **Conference/Venue:** NeurIPS 2023  
- **Abstract:** Inspired by **iterated learning** (compressibility vs. expressivity) and **simplicial embeddings** (approx. discretization), the method resets/relearns representations to induce compressible structure and improves compositional generalization in controlled vision domains and molecular graph tasks, supported by a Kolmogorov-complexity view.
- **Keywords:** iterated learning, simplicial embeddings, compressibility, compositionality.

### [Compositional Generalization from First Principles](https://proceedings.neurips.cc/paper_files/paper/2023/file/15f6a10899f557ce53fe39939af6f930-Paper-Conference.pdf)
- **Year:** 2023  
- **Conference/Venue:** NeurIPS 2023  
- **Abstract:** Treats compositionality as a **property of the data-generating process**. Derives **mild support/architecture** conditions under which models generalize to unseen compositions; validates on synthetic visual setups and relates to identifiable representation learning.
- **Keywords:** data-generating processes, sufficient conditions, identifiable reps, theory + synthetic validation.

### [A Survey on Compositional Generalization in Applications](https://arxiv.org/pdf/2302.01067)
- **Year:** 2023  
- **Conference/Venue:** arXiv preprint  
- **Abstract:** Reviews application-driven settings for compositional generalization, proposing a taxonomy across domains (healthcare, finance, RL, unsupervised/supervised/interactive learning) and trends linking disentanglement/emergent language to practical CG challenges.
- **Keywords:** survey, applications, taxonomy, disentanglement, emergent language.

---

## 2022
### [Lost in Latent Space: Examining Failures of Disentangled Models at Combinatorial Generalisation](local:///mnt/data/NeurIPS-2022-lost-in-latent-space-examining-failures-of-disentangled-models-at-combinatorial-generalisation-Paper-Conference.pdf)

- **Year:** 2022
- **Conference/Venue:** NeurIPS 2022
- **Abstract:** Shows that generative models with highly disentangled latent representations (β-VAE, WAE, SBD-based encoders) still **fail** to generalize to unseen combinations of generative factor values under challenging splits (recombination-to-range). Finds failures arise primarily from **encoder errors**, where novel combinations are mapped to incorrect latent regions. Evaluations on dSprites, 3DShapes, MPI3D, and custom datasets reveal that earlier reports of success relied on overly simple splits.
- **Keywords:** disentanglement, combinatorial generalization, latent-space analysis, encoder/decoder failure modes, generative factors, VAE/WAE.

---

## 2020

### [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://arxiv.org/pdf/1912.09713)
- **Year:** 2020  
- **Conference/Venue:** ICLR 2020  
- **Abstract:** Introduces **DBCA** to construct and assess splits with **similar atom** but **different compound** distributions, and releases **CFQ** (Compositional Freebase Questions). Finds strong **negative correlation** between compound divergence and accuracy across architectures; also builds SCAN splits with DBCA. :contentReference[oaicite:8]{index=8}  
- **Keywords:** CFQ, DBCA, atom vs. compound distributions, benchmark construction.