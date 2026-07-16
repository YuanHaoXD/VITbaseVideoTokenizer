signals are processed as sequences via a dual-head mechanism, employing an autoregressive head for text prediction and a rectified flow matching head for image generation. Extensive experiments confirm that HYDRA achieves superior performance, highlighting the harmony between understanding and generation during joint training. In terms of multimodal understanding, HYDRA outperforms existing native UMMs by an average margin of approximately 10.0 points across eight benchmarks. Meanwhile, it establishes a new benchmark in visual reconstruction with a remarkable rFID of 0.08, providing a robust foundation that facilitates state-of-the-art generation records: 0.86 on GenEval (Ghosh et al., 2023), 86.4 on DPG-Bench (Hu et al., 2024), and 0.53 on WISE (Niu et al., 2025). Additionally, we find that joint training consistently outperforms separate training for both generation and understanding, validating the effectiveness of our harmonized representation. Our contributions are summarized as follows:

• We propose HYDRA-TOK, a representation-harmonized pure ViT that resolves the understanding-generation conflict via a progressive learner, unifying input representations without quantization errors.

- We present HYDRA, a native unified framework that integrates understanding and generation within a single parameter space, utilizing a dual-head mechanism for seamless task execution.

- Empirical results demonstrate that HYDRA outperforms native UMMs by  $ \sim $10.0 points on understanding benchmarks and achieves state-of-the-art performance on GenEval, DPG-Bench, and WISE.

### 2. Method

We present HYDRA, a unified framework harmonizing visual understanding and generation. Central to our approach is HYDRA-TOK, a pure ViT grounded in a key insight: a compact feature space capable of reconstructing inputs serves as a robust foundation for semantic understanding. Adopting a functionally progressive learner design (Fig. 3), we transition continuously from structure-preserving primitives (Gen-ViT) to semantic abstractions (Sem-ViT). This is realized via the Generation-Semantic Bottleneck (GSB) and its novel compress-and-reconstruct operation. By compressing features to filter noise and reconstructing them for distillation, the GSB effectively balances generative fidelity with semantic awareness.

#### 2.1. HYDRA-TOK

Traditional tokenizers face a rigid trade-off between preserving semantic depth and maintaining structural detail. To resolve this issue, HYDRA-TOK reformulates the complete vision transformer backbone into three functionally distinct yet continuous components, effectively followed by a lightweight flow-based decoder.



Generation Vision Transformer (Gen-ViT). Given an input image  $ \mathbf{x} \in \mathbb{R}^{H \times W \times 3} $, we first flatten and project non-overlapping patches into continuous embeddings  $ \mathbf{H}_0 \in \mathbb{R}^{N \times D} $, where  $ N $ and  $ D $ correspond to the number of tokens and dimension. The initial stage, Gen-ViT, is tasked with extracting low-level structural primitives essential for generation. Unlike standard encoders that aggressively compress spatial information, Gen-ViT preserves fine-grained spatial covariance:

 $$ \mathbf{H}_{\mathrm{m i d}}=\mathrm{G e n-V i T}(\mathbf{H}_{0})=\Phi_{L_{\mathrm{g e n}}}\circ\cdots\circ\Phi_{1}(\mathbf{H}_{0}), $$ 

where  $ \Phi_{l} $ denotes the l-th transformer block. This phase ensures that the latent space retains the structural foundation required for high-fidelity synthesis.

Generation-Semantic Bottleneck (GSB). To transition from structure-preserving primitives to semantic abstractions, we introduce the GSB block. Built on a compress-and-reconstruct operation, GSB serves as an information bottleneck that filters extraneous noise to balance the conflicting demands of understanding and generation. As evidenced by our ablation (Fig. 4), while higher dimensions (C) aid reconstruction and understanding, they cause generation performance to collapse (e.g., at  $ C \geq 256 $). This confirms that excessive dimensionality introduces redundancy that disrupts generative stability.

To resolve this, GSB acts as a stabilization pivot by first compressing the intermediate features  $ \mathbf{H}_{\text{mid}} $ into a compact probabilistic space via a lightweight projector  $ \mathbf{W}_{\text{proj}} \in \mathbb{R}^{D \times C} $, where  $ C \ll D $ (typically 64):

 $$ [\mu,\rho]=\mathbf{W}_{\mathbf{p r o j}}\mathbf{H}_{\mathbf{m i d}},\quad\mathbf{z}=\mu+\epsilon\odot\operatorname{e x p}(0.5\rho), $$ 

where  $ \mu, \rho \in \mathbb{R}^{N \times C} $ represent the mean and log-variance, and  $ \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $ is reparameterization noise.

To structure this latent space, we impose a KL divergence loss that aligns the posterior with a standard normal prior:

 $$ \mathcal{L}_{\mathrm{K L}}=-\frac{1}{2}\sum_{j=1}^{C}\left(1+\rho_{j}-\boldsymbol{\mu}_{j}^{2}-\exp(\boldsymbol{\rho}_{j})\right). $$ 

To maintain a coherent flow of information under compression and provide a sufficient foundation for subsequent semantic extraction, we introduce a consistency loss  $ \mathcal{L}_{\mathrm{cos}} $. This loss forces the unprojected features  $ \mathbf{H}_{\mathrm{bn}} = \mu \mathbf{W}_{\mathrm{unproj}}^{\mathrm{und}} $ to maintain directional alignment with the pre-bottleneck features  $ \mathbf{H}_{\mathrm{mid}} $:

 $$ \mathcal{L}_{\mathrm{c o s}}=1-\frac{\mathbf{H}_{\mathrm{m i d}}\cdot\mathbf{H}_{\mathrm{b n}}}{\|\mathbf{H}_{\mathrm{m i d}}\|_{2}\|\mathbf{H}_{\mathrm{b n}}\|_{2}}. $$ 