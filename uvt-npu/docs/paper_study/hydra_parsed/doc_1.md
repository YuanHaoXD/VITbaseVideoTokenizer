<div style="text-align: center;"><img src="imgs/img_in_image_box_159_134_1017_355.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;">(d)</div>


<div style="text-align: center;">Figure 2. Representation schemes in native unified multimodal models. (a) Decoupled Encoder (Deng et al., 2025a; Wu et al., 2025a): It employs a VAE and a representation encoder as dedicated encoders for generation and understanding tasks, respectively. (b) Sequential Encoder (Xie et al., 2025a): It feeds the output of the VAE directly into the representation encoder in a cascaded manner. (c) Single-representation encoder (Ma et al., 2025a; Wu et al., 2025d): It adopts a standalone representation encoder to unify representation learning for both understanding and generation tasks. (d) Our Proposed Representation-Harmonized ViT Design: it also leverages a single ViT backbone, while introducing a bottleneck module to harmonize the feature learning processes of understanding and generation tasks.</div>


### 1. Introduction

Unifying visual understanding and generation has emerged as a pivotal frontier in multimodal intelligence (Deng et al., 2025a; Cao et al., 2025; Liao et al., 2025b). Native unified multimodal models (UMMs) (Wu et al., 2025d; Zhou et al., 2024; Xie et al., 2025a) advocate for direct decoding within a unified parameter space, demonstrating superior synergy over composite UMMs (Ge et al., 2024; Tang et al., 2025; Chen et al., 2025a). However, achieving rational unification is hindered by a fundamental representational divergence, stemming from the fact that understanding and generation constitute inverse tasks with conflicting demands: the former necessitates high-level semantic abstractions, whereas the latter requires compact structural primitives for fine-grained synthesis (Radford et al., 2021; Kingma et al., 2019). This intrinsic conflict forces existing frameworks into disjointed, asymmetric designs, significantly increasing architectural complexity and optimization difficulty.

We attribute this dilemma primarily to the structural limitations of existing image tokenizers, which fail to simultaneously satisfy the three critical criteria illustrated in Fig. 2. First, decoupled paradigms that employ separate encoders for understanding and generation (Deng et al., 2025a; Cao et al., 2025) inherently lack Unification of Input Representation, relying on disjoint features that serve the synergy between the two tasks. Second, sequential architectures that stack representation encoders atop VAEs (Xie et al., 2025a; Liu et al., 2025) theoretically unify the input but compromise the Coherence of Information Flow due to the significant representation mismatch between the generative VAE latent space and the semantic features required by the representation encoder. Third, while utilizing a single shared representation encoder (Ma et al., 2025b; Wu et al., 2025d; Jiao et al., 2025) attempts to solve this, it often suffers from poor Compatibility of Learning Process, where the conflicting objectives of high-frequency detail preservation and semantic abstraction lead to optimization difficulties. Consequently, current methods face an unavoidable trade-off: they either sacrifice generative fidelity, lose semantic alignment, or struggle to converge on a shared representation.



To address this, we propose HYDRA-TOK, a representation-harmonized pure ViT framework. Our core design principle is built on a key insight: a compact feature space capable of reconstructing input data can serve as a robust foundation for semantic understanding. This reconstruction task functions as an information bottleneck, compelling the compact feature to discard extraneous details and instead acquire a vocabulary of dense, structural primitives. These primitives provide a solid basis, thereby enabling the model to construct semantic abstractions from the ground up.

To this end, we reformulate a ViT-based representation encoder (Chen et al., 2024b;c) into a progressive learner that transitions from a Generation vision transformer (GenViT), which captures structure-preserving primitives for high-fidelity synthesis, to a Semantic vision transformer (Sem-ViT). To unify these distinct objectives within a single model, we introduce the Generation-Semantic Bottleneck (GSB). The GSB is architected around a novel compress-and-reconstruct operation, creating an information bottleneck that fosters both high-level semantic abstraction and detailed generative fidelity. Specifically, GSB first compresses features into a compact low-dimension space to filter out noise component, then reconstructs them to the original space for subsequent semantic distillation. In this manner, the compact features can both encode structured details while maintaining semantic awareness.

Built upon HYDRA-TOK, we present HYDRA, a unified framework that achieves complete architectural and representational unification. Leveraging the coherent visual representations provided by HYDRA-TOK, visual