# Vendored FVD 双实现（来源：OmniTokenizer evaluation/common_metrics_on_video_quality/fvd/）。
# - styleganv/: I3D torchscript 版（https://github.com/universome/fvd-comparison），主选（03 篇 §3）
# - videogpt/ : InceptionI3d state_dict 版，并行交叉核对用
# 数值逻辑保持原实现不动；两版的 I3D 权重文件下载到各自子目录（见各自 load_i3d_pretrained）。
