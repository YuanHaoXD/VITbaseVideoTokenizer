<div style="text-align: center;"><img src="imgs/img_in_chart_box_128_146_583_429.jpg" alt="Image" width="37%" /></div>


<div style="text-align: center;">(a) Impact of tokenizer training data size.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_646_146_1076_425.jpg" alt="Image" width="35%" /></div>


<div style="text-align: center;">(b) Layer-wise representational similarity.</div>


<div style="text-align: center;">Figure 8. Analysis of tokenizer data scaling and layer-wise representations. (a) This figure illustrates how understanding (Avg QA), generation (Geneval), and reconstruction (rFID) metrics evolve as the tokenizer training data size increases from 1.2M to 20M. (b) The figure shows the CKNNA score across the 24 layers of the HYDRA-TOK, indicating the change in representational similarity between teacher vit (Chen et al., 2024b).</div>


### C. Additional ablation study results

#### C.1. Decoder Size

We investigate the impact of decoder parameter size on the visual reconstruction quality. Tab. 5 presents a quantitative comparison across three different decoder capacities ranging from approximately 144M to 358M parameters. We observe a clear trend where increasing the decoder size leads to monotonic improvements across all evaluated metrics. Specifically, scaling the decoder from 144.26M to 358.44M results in a 0.99 dB increase in PSNR (from 35.85 to 36.84), a 0.01 improvement in SSIM (from 0.96 to 0.97), and a notable decrease in rFID from 0.17 to 0.14. The largest model variant (358.44M) consistently achieves the best performance, demonstrating that larger decoder capacities are beneficial for achieving high-fidelity visual reconstruction.

<div style="text-align: center;">Table 5. Quantitative comparison of reconstruction quality across different model variants.</div>



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Decoder Size</td><td style='text-align: center; word-wrap: break-word;'>PSNR  $ \uparrow $</td><td style='text-align: center; word-wrap: break-word;'>SSIM  $ \uparrow $</td><td style='text-align: center; word-wrap: break-word;'>rFID  $ \downarrow $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>144.26M</td><td style='text-align: center; word-wrap: break-word;'>35.85</td><td style='text-align: center; word-wrap: break-word;'>0.96</td><td style='text-align: center; word-wrap: break-word;'>0.17</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>240.85M</td><td style='text-align: center; word-wrap: break-word;'>36.55</td><td style='text-align: center; word-wrap: break-word;'>0.96</td><td style='text-align: center; word-wrap: break-word;'>0.15</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>358.44M</td><td style='text-align: center; word-wrap: break-word;'>36.84</td><td style='text-align: center; word-wrap: break-word;'>0.96</td><td style='text-align: center; word-wrap: break-word;'>0.15</td></tr></table>

#### C.2. Scaling HYDRA-TOK training data

For multimodal understanding capabilities, the tokenizer is trained utilizing the LLaVA-1.5 setting (Liu et al., 2023a). As illustrated in Fig. 8a, we evaluate performance across three data regimes: 1.2M, 4M, and 20M image-text pairs. The evaluation metrics include generation (Geneval), reconstruction (rFID), and understanding (Avg QA), where Avg QA denotes the average score across the POPE (Li et al., 2023b), MMBench (Liu et al., 2024a), MMMU (Yue et al., 2024), AI2D (Kembhavi et al., 2016), and RealWorldQA benchmarks. We observe distinct trends for different capabilities as the data size increases. The generative capability, indicated by the General score, shows a consistent positive trend, improving steadily from approximately 38 at 1.2M to over 45 when trained on 20M data. Reconstruction fidelity, measured by rFID (where lower is better), also benefits significantly from larger data scales. While it shows a slight increase at 4M, it achieves its best performance with a sharp drop to approximately 0.08 at the 20M mark. Conversely, multimodal understanding capabilities, as reflected by the Avg QA score, remain relatively stable and robust across all data sizes, maintaining a score consistently above 61. These results suggest that while understanding capabilities are established early, scaling the tokenizer training data is critical for optimizing generation and reconstruction performance.