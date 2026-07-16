### H.2 Image Reconstruction at  $ 1280 \times 768 $

To stress-test generalisation beyond the training resolution, we additionally compare reconstructions at a high resolution of  $ 1280 \times 768 $ and include the dedicated video VAE Wan 2.2 alongside the image-only baselines. This setting exposes how each tokenizer handles dense fine details such as text, foliage, and small structural elements when the spatial token budget is stretched.

<div style="text-align: center;">Input</div>


<div style="text-align: center;">WAN2.2</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_154_307_1022_1215.jpg" alt="Image" width="72%" /></div>


<div style="text-align: center;">Figure 7: Qualitative reconstruction comparison at  $ 1280 \times 768 $. We compare HYDRA-X against Wan 2.2 (Wan et al., 2025), AToken (Lu et al., 2025), and FLUX (Labs et al., 2025).</div>
