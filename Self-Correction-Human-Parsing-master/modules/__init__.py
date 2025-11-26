try:
    from .functions import ACT_RELU, ACT_LEAKY_RELU, ACT_ELU, ACT_NONE
    from .bn import ABN, InPlaceABN, InPlaceABNSync
except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - CPU-only fallback
    import warnings
    import torch.nn as nn
    import torch.nn.functional as F

    warnings.warn(
        "Falling back to pure PyTorch BatchNorm for SCHP InPlaceABN modules. "
        "Performance may differ from the original implementation. "
        f"Original error: {exc}"
    )

    ACT_RELU = "relu"
    ACT_LEAKY_RELU = "leaky_relu"
    ACT_ELU = "elu"
    ACT_NONE = "none"

    class _FallbackABN(nn.Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     activation=ACT_LEAKY_RELU, slope=0.01):
            super().__init__()
            self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
            self.activation = activation
            self.slope = slope

        def forward(self, x):
            x = self.bn(x)
            if self.activation == ACT_RELU:
                return F.relu(x, inplace=True)
            if self.activation == ACT_LEAKY_RELU:
                return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
            if self.activation == ACT_ELU:
                return F.elu(x, inplace=True)
            return x

    ABN = InPlaceABN = InPlaceABNSync = _FallbackABN

from .misc import GlobalAvgPool2d, SingleGPU
from .residual import IdentityResidualBlock
from .dense import DenseModule
