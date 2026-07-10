from .models import register, make, models
from . import transformer
from . import bottleneck
from . import loss
from . import larp_ar
from . import gptc
from . import larp_tokenizer
# === UVT modification: 注册 uvt_tokenizer，使 models.make('uvt_tokenizer') 可实例化（config-driven 训练）===
from .uvt import uvt_tokenizer as _uvt_tokenizer  # noqa: F401  导入即触发 @register


def get_model_cls(name):
    return models[name]