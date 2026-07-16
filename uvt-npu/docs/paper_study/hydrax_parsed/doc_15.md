## A Training Details

### A.1 Tokenizer Training Loss

HYDRA-XTOK is designed as the visual interface of HYDRA-X: before any token reaches the LLM, it must be compact enough for generation, faithful enough for reconstruction, and semantic enough for understanding. We initialize Gen-ViT and Sem-ViT from SigLIP2 (Tschannen et al., 2025). The tokenizer is trained with a reconstruction term and a semantic distillation term:

 $$ \mathcal{L}_{\mathrm{H Y D R A-X T o K}}=\mathcal{L}_{\mathrm{r e c}}+\lambda_{\mathrm{d i s t}}\mathcal{L}_{\mathrm{d i s t}}, $$ 

where  $ \mathcal{L}_{\mathrm{rec}} $ is the reconstruction term detailed below and  $ \mathcal{L}_{\mathrm{dist}} $ aligns Sem-ViT features with the image and video teachers (Ea. 3).

To keep the compact latent both pixel-faithful and structurally stable, the reconstruction term  $ \mathcal{L}_{\mathrm{rec}} $ encapsulates pixel-level recovery, perceptual fidelity, and latent space regularization. Specifically, it combines an L1 loss for direct pixel-space reconstruction, an LPIPS perceptual loss  $ \mathcal{L}_{\mathrm{lpips}} $, an adversarial GAN loss  $ \mathcal{L}_{\mathrm{gan}} $ to refine texture realism, and a Kullback–Leibler (KL) divergence penalty that aligns the posterior with a standard normal prior. The comprehensive reconstruction objective is formulated as:

 $$ \mathcal{L}_{\mathrm{r e c}}=\lambda_{1}\|\mathbf{x}-\hat{\mathbf{x}}\|_{1}+\lambda_{\mathrm{p e r c}}\mathcal{L}_{\mathrm{l p i p s}}+\lambda_{\mathrm{g a n}}\mathcal{L}_{\mathrm{g a n}}-\lambda_{\mathrm{K L}}\sum_{j=1}^{C}\left(1+\boldsymbol{\rho}_{j}-\boldsymbol{\mu}_{j}^{2}-\exp(\boldsymbol{\rho}_{j})\right), $$ 

where x and  $ \hat{x} $ are the original and reconstructed images, while  $ \mu_{j} $ and  $ \rho_{j} $ are the mean and log-variance of the compressed latent.

### A.2 Tokenizer Pre-training

HYDRA-XTOK is trained in three progressive stages to balance foundational representation learning with high-fidelity generative quality:

Stage 1: Foundation Training.  

1.2M at  $ 256 \times 256 $ resolution. We then transition to mixed-resolution training, combining  $ 256 \times 256 $ videos with images ranging from 256 to 2048 pixels. This strategy empowers the tokenizer to generalize effectively to high-resolution video. We optimize the model for 300k iterations using AdamW with a peak learning rate of  $ 2 \times 10^{-4} $, employing a hybrid SigLIP-2 / InternVideo teacher for distillation.

Stage 2: Decoder Refinement. To enhance texture realism and perceptual fidelity, we freeze the encoder and exclusively fine-tune the 27-layer ViT decoder. Adversarial training (GAN loss) is incorporated in this stage to significantly improve visual reconstruction.

Stage 3: Representation Harmonization. In the final stage, we first compute the channel-wise mean and standard deviation of the Gen-ViT latent features. We then freeze Gen-ViT and the decoder while unfreezing Sem-ViT. The Gen-ViT features are normalized before being fed into Sem-ViT and the decoder; during this process, only Sem-ViT is updated. This normalization eliminates feature heterogeneity between the two heads and establishes a unified, semantic-aware latent space capable of faithful reconstruction, which is crucial for downstream UMM tasks.

### A.3 Native Unified Multimodal Models Pre-training

To cultivate the harmonized nature of HYDRA-X, we implement a three-stage progressive training strategy for the unified multimodal model. Detailed configurations and computational cost are summarised in Table 8.

Stage 1: Unified Representation Alignment. To resolve the representation divergence at the input level, we freeze the LLM (Qwen2.5-7B-Instruct) and exclusively tune the vision components (projector, time-step embedding, and flow head). Utilizing 100M image-text pairs, this phase aligns the visual latent space with the linguistic domain, ensuring a coherent unified input representation.

Stage 2: Comprehensive Multimodal Pre-training. We unlock all parameters to facilitate harmonized promotion within a single unified stream. The model is jointly optimized on a balanced mix of 30M understanding samples and 30M generative samples (strategically filtered from Stage 1). We further incorporate approximately 2M image editing samples and 10M video samples into the joint training process. This full-parameter update ensures the compatibility of the learning process and allows the diverse tasks to mutually reinforce each other.