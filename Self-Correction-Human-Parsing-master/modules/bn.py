import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# ===============================================================
# This patched version removes all dependencies on inplace_abn
# ===============================================================

warnings.warn(
    "Using safe PyTorch BatchNorm2d fallback for SCHP InPlaceABN modules. "
    "No custom CUDA extensions are required."
)

# Define activation constants
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"

# Helper: apply activation manually
def _apply_activation(x, activation, slope):
    if activation == ACT_RELU:
        return F.relu(x, inplace=True)
    if activation == ACT_LEAKY_RELU:
        return F.leaky_relu(x, negative_slope=slope, inplace=True)
    if activation == ACT_ELU:
        return F.elu(x, inplace=True)
    return x


class ABN(nn.Module):
    """Activated Batch Normalization (safe PyTorch version).

    This mirrors the parameter structure of the original InPlaceABN module so
    pre-trained checkpoints with keys such as ``bn.weight`` can still be loaded.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        activation="leaky_relu",
        slope=0.01,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.activation = activation
        self.slope = slope

        self.bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=True,
        )

    @property
    def weight(self):
        return self.bn.weight

    @weight.setter
    def weight(self, value):
        with torch.no_grad():
            self.bn.weight.copy_(value)

    @property
    def bias(self):
        return self.bn.bias

    @bias.setter
    def bias(self, value):
        with torch.no_grad():
            self.bn.bias.copy_(value)

    @property
    def running_mean(self):
        return self.bn.running_mean

    @running_mean.setter
    def running_mean(self, value):
        with torch.no_grad():
            self.bn.running_mean.copy_(value)

    @property
    def running_var(self):
        return self.bn.running_var

    @running_var.setter
    def running_var(self, value):
        with torch.no_grad():
            self.bn.running_var.copy_(value)

    def forward(self, x):
        x = self.bn(x)
        return _apply_activation(x, self.activation, self.slope)

    def __repr__(self):
        rep = (
            f"{self.__class__.__name__}({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"activation={self.activation}"
        )
        if self.activation == "leaky_relu":
            rep += f", slope={self.slope})"
        else:
            rep += ")"
        return rep


class InPlaceABN(ABN):
    """Safe replacement for InPlaceABN using standard BatchNorm2d"""

    def forward(self, x):
        # identical behavior without CUDA extension
        return super().forward(x)


class InPlaceABNSync(ABN):
    """Safe replacement for InPlaceABNSync without CUDA dependency"""

    def forward(self, x):
        # simply run the normal ABN forward pass
        return super().forward(x)
