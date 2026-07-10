# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**UVT (Unified Video Tokenizer)** — a research project building a single ViT (SigLIP2-So400M initialized) that is simultaneously: image+video unified (1 anchor frame + 16-frame clip protocol), reconstruction+semantic unified (high-fidelity pixel recon *and* linearly-readable semantics), and generatable (a DiT trains cleanly on the frozen latent). Borrowed from Hydra-X's three designs (tubelet causal attention / hierarchical temporal patchify / Decompressor dual-teacher distillation), deliberately narrowed to "tokenizer only, no UMM". Read `docs/08` first for the mental model, then `docs/01`–`05` for architecture/code/data/experiments/task-book.

## Two repos, two environments (do not cross-wire)

- **`uvt/`** — the main repo (forked from LARP, MIT). Real training happens here. Entry is `train.py` (torchrun, bare DDP, no Lightning).
- **`phase-b-omnitokenizer/`** — the Phase B boundary-condition experiment repo (forked from OmniTokenizer, MIT). **Runs in its own locked docker** (PL 1.5.4, torch 2.2.1/cu118). It is a scientific control asking whether Hydra-X's "less is more" findings survive without pretraining priors — forked OmniTokenizer with two added knobs: `--temporal_attn_mode {tubelet,causal,full}` and `--temporal_fold_mode {single,learned,avgpool}`. See `phase-b-omnitokenizer/docs/phase-b-changes.md`.

Binaries (`*.pt/*.ckpt/*.pth/*.bin/*.safetensors`) and datasets are gitignored — download per `docs/07`. SigLIP2-So400M-patch16-**256** (`google/siglip2-so400m-patch16-256`) is a hard prerequisite: it is "one weight, three uses" (Gen/Sem-ViT + decoder init, frozen image teacher, zero-shot text tower).

## Commands

All `uvt/` commands run **from inside the `uvt/` directory** — tests and code use top-level package imports (`from models.uvt...`, `from losses...`, `from teachers...`).

### Tests (~84 pass on CPU; no GPU, no weight downloads)
```bash
cd uvt
pytest tests/ -v                                   # full suite
pytest tests/test_smoke_train_step.py -v           # single file
pytest tests/test_tokenizer.py -k test_name -v     # single test
```
Tests use `UVTConfig(tiny=True)` / `SigLIP2Teacher(tiny=True)` to build tiny offline models (no SigLIP2 download, no `decord`/CUDA). GPU/decord-dependent paths skip via `importorskip`. The capstone integration test is `tests/test_smoke_train_step.py` (one image + one video train step end-to-end).

### Training (torchrun entry, `uvt/train.py`)
The entry reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars (torchrun-injected) and degrades to single-process when they're absent (e.g. `python train.py` for CPU debugging). Config is `--cfg <yaml>`; scalars flow in two ways:
- **`$var$` substitution** — yaml fields like `frame_num: $frame_num$` are replaced from the matching CLI arg (`--frame_num 17`).
- **`--opts dotted.path value`** — deep-merge overrides into the config tree, e.g. `--opts stage 2`, `--opts teachers.vid_mock true`, `--opts model.args.tiny true`.

```bash
# null smoke (P1-smoke gate) — LARP fake-data mechanism, no dataset download
torchrun --nproc_per_node=8 train.py --cfg cfgs/uvt_stage1.yaml \
    --csv_file null128 --batch_size 2 --frame_num 17 --input_size 256 \
    --opts teachers.vid_mock true

# Stage 1 real training; switch stages with --opts stage {1,2,3}
torchrun --nproc_per_node=8 --nnodes=N train.py --cfg cfgs/uvt_stage1.yaml \
    --csv_file <imagenet_train.csv> --batch_size 32 --frame_num 17 --input_size 256 \
    --opts teachers.vid_mock false
```
OOM → `--opts grad_accumulates 8` (global batch unchanged). `--csv_file null128` triggers LARP's fake-data path; `vid_mock=true` substitutes a `MockTeacher` for the video teacher (avoids downloading InternVideo).

### Eval
`eval/protocols.py` is the **single source of truth for all eval preprocessing** (interpolation kernel/crop must not drift). Suites: `eval/recon_metrics.py` (PSNR/SSIM/rFID/rFVD), `eval/semantic/{zeroshot,linear_probe,cknna}.py`, `eval/calibrate.py` (five-anchor calibration). CKNNA's reference model must be DINOv2 (third-party), **never the teacher itself** (circular reasoning).

### Phase B
Separate docker. `python vqgan_train.py --tokenizer omnitokenizer --temporal_attn_mode <MODE> --temporal_fold_mode <FOLD> ... --seed 1`. Full run matrix and the compression-ratio caveat in `phase-b-omnitokenizer/docs/phase-b-changes.md` §4.

## Architecture: the single-ViT pipeline

`uvt/models/uvt/uvt_tokenizer.py` (`UVTTokenizer`, registered `@register('uvt_tokenizer')`) assembles the parts loaded from a sliced SigLIP2 (27 layers split 13/14):
```
x [B,3,17,256,256]
 → space patchify 16× (SigLIP2) → temporal Fold-2× (anchor isolated) → [B,1+8,256,1152]
 → Gen-ViT front (layers 1–6, tubelet mask) → temporal Fold-2× → [B,1+4,256,1152]
 → Gen-ViT back (layers 7–13) → h → GSB.compress → z [B,1+4,256,64]  (the only outward latent)
     ├─ PixelDecoder (27-layer symmetric ViT + 2× TemporalUnfold) → x_hat   (recon: L1+LPIPS+KL, +GAN in S2)
     └─ Sem-ViT (layers 14–27) + MAP head → s, s_pool
            ├─ s[:,0]+s_pool ──cosine──► SigLIP2 teacher (patch+pool)
            └─ s[:,1:] ► Decompressor(4× temporal upsample, train-only, discarded) ► InternVideo teacher
```
Compression: space 16×, time 4×, channel 1152→64. Part map: M-1…M-10 = `attention_mask`/`blocks`/`gsb`/`siglip_backbone`/`temporal_fold`/`encoder`(GenViT)/`sem_vit`/`decoder`(PixelDecoder)/`decompressor`/`uvt_tokenizer`; L-1..3 = `losses/{recon,distill,gan}`; T-1..2 = `teachers/{siglip2,internvideo}_teacher`; D-1..3 = `datasets/{video,image,joint}_...`; TR-1..2 = `trainers/{base_trainer,uvt_tokenizer_trainer}`; E-1..4 = `eval/*`.

### Three-stage state machine (`TR-2`, stage set via `--opts stage N`, freezing enforced by `UVTTokenizer.set_stage`)
- **Stage 1 (base):** all params trainable; L1+LPIPS+KL+L_cos recon (L-1) + λ·distill (L-2).
- **Stage 2 (refine):** only decoder trainable (encoder/GSB frozen); recon with **L_cos explicitly off** + GAN (L-3, LARP recipe). G/D alternate; `d_update_freq` counts in **optimizer steps** (whole grad-accum window opens/closes together).
- **Stage 3 (harmonize):** first run `model.estimate_latent_stats` (ADR-5: collect z channel mean/std into GSB buffers, set `normalize=True`), then train **only Sem-ViT** + distill.

### Contracts that are easy to silently break (from `uvt_tokenizer.py` docstring + `docs/08 §7`)
- **`forward` → `forward_train` dispatch (contract ③):** DDP only hooks `forward`; in training mode it must delegate to `forward_train` or gradients won't sync. Don't bypass it.
- **normalize three-way split (Stage-3 crux):** decoder consumes **physical** `z`; Sem-ViT consumes **canonical** `gsb.to_canonical(z)`; `L_cos`'s `mu_proj = sem_vit.in_proj(to_canonical(mu))`. When `normalize=False` (S1/S2) `to_canonical` is identity. Some early `docs/01`/`05` text describing this is **stale — code wins**.
- **`gsb` has no `expand`/`unproj`** (contract ④): back-projection lives in each consumer's own `in_proj`. Never call `gsb.expand`.
- **Decompressor is video-only** (contract ⑥): `forward_train` builds `decomp_out` only when `F>1`; image batches get `None` and L-2's vid term masks off.
- **Decompressor is stripped from checkpoints** via a `register_state_dict_post_hook` (train-only attachment); distill heads + GAN discriminator live in `self.loss` (an `nn.ModuleDict`), so they go into the *loss* segment of checkpoints, not the *model* segment — `model_sd_only` checkpoints stay clean.

### Config → model wiring
`models.make('uvt_tokenizer', args)` instantiates `UVTTokenizer(cfg=None, **kwargs)` — yaml `model.args` are passed as kwargs into `UVTConfig`. `trainers.trainers_dict['uvt_tokenizer_trainer']` resolves the trainer. Both registries are populated by import side-effects in `models/__init__.py` and `trainers/__init__.py`.

## Discipline (red lines — see `docs/08 §9`)

- **Docs are authoritative.** `docs/01`–`05` settle architecture disputes; `docs/07` governs running; `docs/08` is the mental-model + lessons-learned handover (read it first). Where `docs/01`/`05` text conflicts with code, **code + `docs/08 §7` win** (several contracts were revised during implementation and the task-book text wasn't updated).
- **Frozen interfaces.** Class names/signatures/shapes in `docs/05` task cards are a contract. Don't silently change them — propose a revision and record it in `docs/06 §6`.
- **ADR decisions are config fields, never hardcoded.** Ambiguous decisions (fold_positions, attn_mode, rope_dims, kl_weight…) live on `UVTConfig` with an `ADR-#` comment.
- **Banned:** pytorch-lightning and `flex_attention` in the main repo (attention masks use SDPA additive bias). Phase B *is* Lightning, but only inside its locked docker — never touch the PL version.
- **Paired ablations (D13):** variants reuse the same seed/init/data-order and compare pairwise deltas. `JointLoader`'s `sampling_trace.json` guarantees that determinism — don't break it.
- **Statistical discipline:** ablations ≥2 seeds, same data order; report delta-sign consistency; report negative results honestly.
- **License:** LARP / OmniTokenizer / LeanVAE are MIT (forkable). AToken is Apple Sample Code — read-only reference, **never commit**.

## Known forward-compat shims & gotchas (14 real bugs already fixed — `docs/08 §6`, `docs/07 §5`)

These are fixed; the point is to recognize the *class* of issue when deps drift:
- `transformers` 5.13 removed `SiglipVisionModel.vision_model` and changed `get_text_features` → return ModelOutput → shims in `siglip_backbone._tower()` and `siglip2_teacher._text_features_as_tensor()`. **Recommend pinning `transformers>=4.47,<6`.**
- scipy removed `sqrtm(disp=)` → vendored `eval/fid/fid_score.py` uses `inspect.signature` fallback.
- When editing `cfgs/*.yaml`: keep the `train_dataset` node (LARP's `make_cfg` preprocess needs it even though `TR-2` actually loads via `joint_dataset`); `joint_dataset.sources` must be the nested `{dataset:{name,args}}` form; `$var$` inside lists *is* now substituted (a list-recursion fix).
- `base_trainer` has ~8 additive `torch.cuda.is_available()` guards so it runs on CPU for dev-machine validation — **these are intentional, not bugs; the GPU path is unchanged.**

## Doc map (read which for what)

| Doc | When to read |
|---|---|
| `docs/08` | **First** — mental model + history of 14 bugs + forward tasks |
| `docs/01` | Before changing the model (full spec + tensor flow + ADRs) |
| `docs/02` | Before changing code (repo/env/per-file change list + unit tests) |
| `docs/03` | Before running eval (data pipeline + the single eval protocol + 5-anchor calibration) |
| `docs/04` | Before designing experiments (per-experiment enum + Gates + stat discipline + risk register) |
| `docs/05` | Before implementing any module (per-file task cards, interfaces frozen to signature) |
| `docs/07` | First time running on the server (env/weights/data/commands) |
| `docs/损失详情.md` | When tuning/diagnosing losses (what each loss is/why/runaway symptoms) |
| `docs/background/` | Deprecated early research — context only, superseded by `docs/01`–`08` |
