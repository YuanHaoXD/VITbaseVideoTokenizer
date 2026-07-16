<div style="text-align: center;"><img src="imgs/img_in_chart_box_111_135_340_304.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(a) Generation.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_350_135_577_302.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(b) Understanding.</div>


<div style="text-align: center;">Figure 7. Analysis of representation-harmonized co-promotion. (a) For generation, joint training enhances generation efficiency by stabilizing the latent space. (b) For understanding, joint training surpasses single-task baselines after a crossover point, showing that generative constraints refine perceptual precision. Details of each understanding benchmark and experiment setting is shown in Fig. 10 and Appendix D.2.</div>


throughout its pure ViT backbone. At the 1.5B scale, HYDRA demonstrates dominant performance with an average score of 63.1, surpassing the strong baseline Show-o2 (53.2) by a significant margin. This advantage is particularly pronounced in fine-grained tasks sensitive to information loss, such as OCRBench (Liu et al., 2024b), where HYDRA achieves 50.8, a score more than double that of Show-o2 (24.5). Scaling up to 7B, HYDRA continues to lead, outperforming Show-o2 on complex reasoning benchmarks like MMStar (Chen et al., 2024a) (62.3 vs. 56.6) and SEED (Li et al., 2023a) (75.5 vs. 69.8). Most notably, in OCRBench (Liu et al., 2024b), HYDRA preserves high-frequency character details often discarded by VAE-based unified models, scoring 57.7 compared to 32.4 for Show-o2.

RQ4: Compatibility of Generation Learning Process. We begin by investigating the mechanism behind this compatibility through the training dynamics illustrated in Fig. 8. First, regarding generative harmony (Fig. 7a), we observe that joint training (U&G) consistently outperforms single-task generation (G Only). It not only demonstrates higher convergence efficiency in the early stage (Phase 1) but also achieves superior final fidelity (Phase 2). This validates that the semantic alignment provided by understanding tasks effectively stabilizes the generative latent space, leading to faster and better convergence. Second, regarding understanding harmony (Fig. 7b), a distinct trend appears. While single-task understanding (U Only) learns faster initially, joint training (U&G) overtakes it after a critical crossover point (~6k steps). This phenomenon demonstrates that although generation tasks are harder to optimize initially, their fine-grained structural constraints eventually refine the model's perceptual precision, proving that generation and understanding are mutually reinforcing in our framework.

Supported by this harmonized training dynamic, we further validate the learning compatibility on text-to-image generation benchmarks in Tab. 3. A common failure mode in UMMs is the “tug-of-war” where improving understanding degrades generation. HYDRA breaks this trade-off, demonstrating that joint optimization leads to superior generative performance. At the 1.5B scale, HYDRA establishes a new benchmark for native UMMs, achieving an Overall GenEval score of 0.86 and DPG-Bench score of 85.51, significantly outperforming Show-o2 (0.73 / 85.02). At the 7B scale, HYDRA sets new state-of-the-art records with a GenEval Overall score of 0.86, surpassing both the unified model Ming-UniVision (Huang et al., 2025) (0.85) and the specialized 12B model FLUX.1 [Dev] (0.82) (Labs et al., 2025). Furthermore, on the WISE benchmark, HYDRA achieves an overall score of 0.53, demonstrating robust alignment across diverse cultural and spatial contexts compared to existing native unified baselines.



#### 3.2. Ablation Analysis

HYDRA-TOK Training Objectives. Fig. 6 validates our tokenizer's training objectives. The baseline, relying solely on reconstruction loss ( $ L_{rec} $), suffers feature collapse and yields suboptimal results. Teacher initialization provides a crucial semantic scaffold, boosting performance, while distillation ( $ L_{dist} $) further enhances comprehension without compromising structure. Ultimately, the complete objective achieves optimal synergy, balancing understanding, generation, and reconstruction to robustly support the unified architecture.

HYDRA Training Stages. Tab. 4 confirms the indispensable ability of each stage in our progressive training recipe. Omitting Stage I causes a universal performance decline, underscoring its role in initial alignment. Removing Stage II severely degrades generation while impairing understanding, proving it critical for consolidating the unified feature space. Skipping Stage III results in a total loss of instruction-following capabilities for QA tasks. Consequently, the full recipe yields peak performance across all benchmarks.

### 4. Conclusion

In this work, we present HYDRA-TOK and HYDRA to reconcile the intrinsic conflict between visual understanding and generation. By employing the Generation-Semantic Bottleneck, HYDRA-TOK functions as a progressive learner that harmonizes structural primitives with semantic abstractions, effectively achieving the Unification of Input Representation. This cohesive foundation enables HYDRA to integrate understanding and generation within a single parameter space, satisfying the critical criteria of Coherence of Information Flow and Compatibility of Learning Process. Our extensive experiments not only establish new state-of-the-art benchmarks but also reveal a fundamental insight: understanding and generation are not competitive objectives.