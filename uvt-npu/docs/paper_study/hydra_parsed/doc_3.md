<div style="text-align: center;"><img src="imgs/img_in_image_box_162_140_1025_538.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">(a) HYDRA-Tok</div>


<div style="text-align: center;">(b) HYDRA</div>


<div style="text-align: center;">Figure 3. Training process illustration for HYDRA-TOK and HYDRA. (a) HYDRA-TOK functions as a progressive learner, bridging the gap between reconstruction and understanding. It employs a Generation-Semantic Bottleneck (GSB) to execute a unique compress-and-reconstruct operation, effectively filtering noise to transition from structure-preserving primitives (Gen-ViT) to semantic abstractions (Sem-ViT). (b) HYDRA achieves representational unification upon this foundation, utilizing a dual-head mechanism to seamlessly integrate autoregressive text prediction with rectified flow matching for image generation.</div>


Based on our experiments, we set the regularization weights to  $ \lambda_{\mathrm{KL}} = 10^{-4} $ and  $ \lambda_{\mathrm{cos}} = 1.0 $. The composite bottleneck objective is defined as  $ \mathcal{L}_{\mathrm{reg}} = \lambda_{\mathrm{KL}} \mathcal{L}_{\mathrm{KL}} + \lambda_{\mathrm{cos}} \mathcal{L}_{\mathrm{cos}} $.

Semantic Vision Transformer (Sem-ViT). As the final stage of our functionally progressive learner, Sem-ViT acts as a deep non-linear mapper. Its primary role is to transition the structural foundations (encoded in  $ H_{bn} $) into a high-dimensional semantic space, thereby achieving rational disentanglement:

 $$ \mathbf{H_{o u t}}=\operatorname{S e m-V i T}(\mathbf{H_{b n}})=\Phi_{L_{\mathrm{t o t a l}}}\circ\cdots\circ\Phi_{L_{\mathrm{g e n}}+1}(\mathbf{H_{b n}}). $$ 

To ensure robust representation learning across the entire hierarchy, we employ semantic self-distillation on both GenViT and Sem-ViT. We align the intermediate features from distinct depths of the student with a frozen, pre-trained ViT (Chen et al., 2024b) via cosine similarity maximization:

 $$ \mathcal{L}_{\mathrm{d i s t}}=\sum_{l\in\mathcal{S}_{\mathrm{g e n}}\cup\mathcal{S}_{\mathrm{g e m}}}\left(1-\frac{\mathbf{H}^{(l)}(\mathbf{x})\cdot\mathcal{T}^{(l)}(\mathbf{x})}{\|\mathbf{H}^{(l)}(\mathbf{x})\|_{2}\|\mathcal{T}^{(l)}(\mathbf{x})\|_{2}}\right). $$ 

where  $ S_{gen} $ and  $ S_{sem} $ denote the selected layers from GenViT and Sem-ViT, respectively.

Pixel Flow Decoder. To unburden the backbone, we employ a lightweight decoder  $ \mathbf{v}_{\theta} $ that utilizes flow matching to recover high-frequency details. Conditioned on latent c, it learns to regress the velocity field by minimizing:

 $$ \begin{array}{r}{\mathcal{L}_{\mathrm{F M}}=\mathbb{E}_{t,\mathbf{x},\epsilon}\left[\|\mathbf{v}_{\theta}(\mathbf{x}_{t},t,\mathbf{c})-(\epsilon-\mathbf{x})\|^{2}\right].}\end{array} $$ 

To further enhance perceptual fidelity, we enforce an LPIPS loss  $ \mathcal{L}_{1\mathrm{pips}} $ on the estimated clean image  $ \hat{\mathbf{x}} $ and incorporate an adversarial GAN loss  $ \mathcal{L}_{\mathrm{gan}} $ to refine texture realism. We empirically set  $ \lambda_{\mathrm{FM}} = 1.0 $,  $ \lambda_{\mathrm{perc}} = 0.1 $, and  $ \lambda_{\mathrm{gan}} = 0.075 $. The total reconstruction loss is formulated as  $ \mathcal{L}_{\mathrm{rec}} = \lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{perc}}\mathcal{L}_{1\mathrm{pips}} + \lambda_{\mathrm{gan}}\mathcal{L}_{\mathrm{gan}} $.

Total objective. Finally, the unified tokenizer is optimized by minimizing the weighted sum of the reconstruction, regularization, and alignment objectives defined above:

 $$ \mathcal{L}_{\mathrm{t o k e n i z e r}}=\mathcal{L}_{\mathrm{r e c}}+\mathcal{L}_{\mathrm{r e g}}+\lambda_{\mathrm{d i s t}}\mathcal{L}_{\mathrm{d i s t}}, $$ 

where the distillation weight is set to  $ \lambda_{dist} = 1.0 $ by default.

#### 2.2. HYDRA

Built upon the robust representations of HYDRA-TOK, HYDRA represents a native unified framework that integrates understanding and generation within a single parameter space. By leveraging the coherent visual representations from our tokenizer, HYDRA processes visual and textual signals as a unified sequence via a shared autoregressive transformer, employing a specialized dual-head mechanism to reconcile their distinct output modalities.

Unified Input Representation. To achieve the Unification of Input Representation, we integrate visual signals into the LLM by treating them as continuous sequences fully compatible with the embedding space. For a given image, we extract its latent representation  $ H_{bn} $ from the GSB. We