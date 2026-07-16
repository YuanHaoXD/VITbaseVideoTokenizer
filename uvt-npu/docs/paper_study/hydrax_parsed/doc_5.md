<div style="text-align: center;"><img src="imgs/img_in_image_box_145_143_1042_733.jpg" alt="Image" width="75%" /></div>


<div style="text-align: center;">Figure 4: HYDRA-X unifies five visual tasks through the holistic tokenizer HYDRA-XTOK. (a) HYDRA-XTOK encodes any image or video into a compact Gen-ViT latent and then into semantic features with Sem-ViT. (b) Previous editing pipelines (left) encode source and target with two independent branches; HYDRA-X (right) keeps Gen-ViT independent for faithful reconstruction but shares the Sem-ViT with tubelet causal attention, injecting structural interaction inside the tokenizer. (c) A shared backbone with two separate heads drives all five tasks.</div>


Sem-ViT to bidirectional attention uniformly degrades every metric, mirroring F1: less attention is more even on the understanding side.

- A semantic latent lifts both understanding and generation. Dual image- and video-teacher distillation, enabled by the Decompressor, equips the compact latent with explicit spatiotemporal semantic structure, jointly improving understanding and generation.

## 5 HYDRA-X: Advancing Unified Multimodal Models with Holistic Tokenizers

### 5.1 Overall Architecture

HYDRA-X follows the standard native UMM template (Xie et al., 2025a; Liu et al., 2025b; Qiu et al., 2026): text tokens and visual tokens produced by HYDRA-XTOK are interleaved into a single sequence and processed by a shared LLM backbone with two specialised heads, an autoregressive language head trained with next-token prediction and a vision head trained with flow matching (Lipman et al., 2022; Esser et al., 2024). Within this template, HYDRA-X unifies five tasks under one shared tokenizer HYDRA-XTOK (Fig. 4(a)): image generation (text  $ \rightarrow $ image), image understanding (image  $ \rightarrow $ text), video generation (text  $ \rightarrow $ video), video understanding (video  $ \rightarrow $ text), and image editing (source image with text instruction  $ \rightarrow $ target image).

As illustrated in Fig. 4(c), the same Gen-ViT serves all five tasks; the only task-dependent component is which head decode the LLM output. The model is trained end-to-end with the composite loss

 $$ \begin{array}{c} \mathcal{L}_{HYDRA-X}~=~\lambda_{1}\mathcal{L}_{NTP}+\lambda_{2}\mathcal{L}_{FM}, \end{array} $$ 

where  $ \mathcal{L}_{\mathrm{NTP}} $ is the next-token prediction loss for text,  $ \mathcal{L}_{\mathrm{FM}} $ is the rectified flow matching loss for visual latents, and both loss weights  $ \lambda_1 $ and  $ \lambda_2 $ are set to 1 by default.