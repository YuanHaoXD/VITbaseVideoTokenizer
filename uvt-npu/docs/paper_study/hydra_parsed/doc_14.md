<div style="text-align: center;"><img src="imgs/img_in_image_box_267_137_920_690.jpg" alt="Image" width="53%" /></div>


<div style="text-align: center;">Figure 9. Qualitative comparison of t-SNE.</div>


#### C.3. Visualization of CKNNA

To better illustrate the coherence of information flow within our model, we conducted a representational similarity analysis using Centered Kernel Nearest-Neighbor Alignment (CKNNA) (Huh et al., 2024). We randomly selected 10,000 images from the ImageNet2012 validation set (Russakovsky et al., 2014) to calculate the CKNNA metric between our HYDRA-TOK and the teacher model, InternViT. As observed in Fig. 8b, our model exhibits strong alignment in early layers.

Simultaneously, we calculated the CKNNA index between the generation features and understanding features within our model. Our model achieves a score of 0.10, which is significantly higher than the 0.03 achieved by the Show-o2 model. This indicates a highly coherent transition between generation and understanding feature representations in our approach. Furthermore, we observe that as unified training progresses, this metric continuously increases, eventually reaching 0.13.

#### C.4. Visualization of t-SNE

As shown in Fig. 9, we visualize the learned features from both the generation and understanding branches of HYDRA, comparing them against UniFlow (Yue et al., 2025) and Show-o2 (Xie et al., 2025a) (equipped with WAN (Wan et al., 2025) and SigLIP (Tschannen et al., 2025)). While the baselines often exhibit disparately distributed features, HYDRA demonstrates distinct class clusters in both its generation and understanding representations. This strong semantic discriminability indicates a high degree of alignment between the two feature spaces. Such similarity confirms that our architecture establishes a coherent information flow, enabling the two tasks to be collaboratively optimized within a harmonized representational framework.

### D. Training details

HYDRA-TOK is training in two progressive stages to effectively balance representation learning and generative quality. Stage 1: Foundation Training. In the initial stage, we focus on establishing the foundational capabilities of the tokenizer through joint reconstruction and distillation objectives. We utilize a large-scale composite dataset consisting of ImageNet-1.2M, CC-12M, and SAM-10M. During this phase, the entire tokenizer (both encoder and decoder) is trained end-to-end. We optimize the model for a total of 300k iterations with a global batch size of 256. The learning rate is set to  $ 2e^{-5} $.