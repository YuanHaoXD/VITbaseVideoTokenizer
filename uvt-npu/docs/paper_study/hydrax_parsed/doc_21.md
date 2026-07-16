## H Qualitative Comparisons

This section presents qualitative results across the five tasks supported by HYDRA-X. We compare against representative baselines drawn from both unified multimodal models and task-specialised systems, and organise the comparisons by task and resolution.

### H.1 Image Reconstruction at  $ 512 \times 512 $

We first inspect reconstruction fidelity at the standard  $ 512 \times 512 $ resolution. The comparison spans three families of baselines: dedicated image VAEs (FLUX), unified tokenizers built into UMMs (MingTok, AToken), and the recently proposed RAE. The visual difference makes texture, fine-edge, and small-text fidelity directly comparable.

<div style="text-align: center;"><img src="imgs/img_in_image_box_152_419_1037_1120.jpg" alt="Image" width="74%" /></div>


<div style="text-align: center;">Figure 6: Qualitative reconstruction comparison at  $ 512 \times 512 $. We compare HYDRA-X against RAE (Zheng et al., 2025), MingTok (Huang et al., 2025), AToken (Lu et al., 2025), and FLUX (Labs et al., 2025).</div>
