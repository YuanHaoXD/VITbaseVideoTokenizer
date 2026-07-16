# HYDRA: Unifying Multi-modal Generation and Understanding via Representation-Harmonized Tokenization

Xuerui Qiu $ ^{*123} $ Yutao Cui $ ^{*2} $ Guozhen Zhang $ ^{*4} $ Junzhe Li $ ^{5} $ JiaKui Hu $ ^{5} $ Xiao Zhang $ ^{2} $ Yang Li $ ^{2} $ Songtao Liu $ ^{2} $ Miles Yang $ ^{2} $ Yu Shi $ ^{3} $ Zhao Zhong $ ^{2} $ Liefeng Bo $ ^{2} $

<div style="text-align: center;"><img src="imgs/img_in_chart_box_111_430_470_785.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;">(a) Comparison with SOTA.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_488_436_1071_786.jpg" alt="Image" width="47%" /></div>


(b) Image generation results.

<div style="text-align: center;">Figure 1. Multimodal understanding and image generation results from HYDRA. (a) Our model outperforms previous state-of-the-art unified multimodal models as well as several task-specific models across diverse benchmarks. (b) Our model demonstrates robust visual generation capabilities, producing high-fidelity images with accurate semantic alignment.</div>


## Abstract

Unified Multimodal Models struggle to bridge the fundamental gap between the abstract representations needed for visual understanding and the detailed primitives required for generation. Existing approaches typically compromise by employing decoupled encoders, stacking representation encoder atop VAEs, or utilizing discrete quantization. However, these methods often disrupt information coherence and lead to optimization conflicts. To this end, we introduce HYDRA-TOK, a representation-harmonized pure ViT in the insight that visual modeling should evolve from gener

tion to understanding. HYDRA-TOK reformulates the standard backbone into a progressive learner that transitions from a Gen-ViT, which captures structure-preserving primitives, to a SemViT for semantic encoding. Crucially, this transition is mediated by a Generation-Semantic Bottleneck (GSB), which compresses features into a low-dimensional space to filter noise for robust synthesis, then restores dimensionality to empower complex semantic comprehension. Built upon this foundation, we present HYDRA, a native unified framework integrating perception and generation within a single parameter space. Extensive experiments establish HYDRA as a new state-of-the-art. It sets a benchmark in visual reconstruction (rFID 0.08) and achieves top-tier generation performance on GenEval (0.86), DPG-Bench (86.4), and WISE (0.53), while simultaneously outperforming previous native UMMs by an average of  $ \sim $10.0 points across eight challenging understanding benchmarks.