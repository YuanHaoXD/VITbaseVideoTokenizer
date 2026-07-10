from .trainers import register, trainers_dict
from . import base_trainer, larp_ar_trainer, larp_tokenizer_trainer
from . import larp_ar_fp_trainer
# === UVT modification (TR-2): 注册三阶段状态机训练器 ===
from . import uvt_tokenizer_trainer
# === UVT modification end ===
