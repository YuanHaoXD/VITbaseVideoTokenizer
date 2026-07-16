from .datasets import register, make
from . import video_dataset
# === UVT modification (D-2/D-3): 注册图像数据集与联合 loader ===
from . import image_dataset
from . import parquet_image_dataset   # D-2b: HF parquet 直读(ImageNet-1k)
from .joint_loader import JointLoader
# === UVT modification end ===
