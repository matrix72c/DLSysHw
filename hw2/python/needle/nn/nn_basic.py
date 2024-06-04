"""The module.
"""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(self.in_features, self.out_features)
        )
        self.bias = (
            Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1)))
            if bias
            else None
        )

    def forward(self, X: Tensor) -> Tensor:
        out = ops.matmul(X, self.weight)
        if self.bias:
            out = ops.add(out, ops.broadcast_to(self.bias, out.shape))
        return out


class Flatten(Module):
    def forward(self, X):
        if len(X.shape) == 1:
            return X
        n = 1
        for i in X.shape[1:]:
            n *= i
        return ops.reshape(X, (X.shape[0], n))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        softmax = ops.LogSumExp(axes=1)(logits)
        I_y = init.one_hot(logits.shape[1], y)
        z = softmax - ops.summation(logits * I_y, axes=1)
        return ops.summation(z) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = Tensor(init.zeros(dim), device=device, dtype=dtype)
        self.running_var = Tensor(init.ones(dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(self.weight, x.shape)
        broadcast_bias = ops.broadcast_to(self.bias, x.shape)
        if self.training:
            # E[x]
            mean_x_batch = ops.summation(x, axes=0) / batch_size
            mean_x_batch = ops.reshape(mean_x_batch, (1, -1))
            broadcast_mean_x_batch = ops.broadcast_to(mean_x_batch, x.shape)

            # Var[x]
            var_batch = (x - broadcast_mean_x_batch) ** 2
            var_batch = ops.summation(var_batch, axes=0) / batch_size
            var_batch = ops.reshape(var_batch, (1, -1))
            std_dev_batch = (var_batch + self.eps) ** (0.5)
            broadcast_std_dev_batch = ops.broadcast_to(std_dev_batch, x.shape)

            # Normalize
            x_norm = broadcast_weight * (x - broadcast_mean_x_batch) / broadcast_std_dev_batch + broadcast_bias

            # Update running mean and var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x_batch
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
        else:
            broadcast_running_mean = ops.broadcast_to(self.running_mean, x.shape)
            broadcast_running_var = ops.broadcast_to(self.running_var, x.shape)
            x_norm = broadcast_weight * (x - broadcast_running_mean) / (broadcast_running_var + self.eps) ** (0.5) + broadcast_bias
        return x_norm


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        mean = ops.broadcast_to(
            ops.reshape(ops.summation(x, axes=1) / x.shape[1], (batch, 1)), x.shape
        )
        var = ops.broadcast_to(
            ops.reshape(
                ops.summation(ops.power_scalar(x - mean, 2), 1) / self.dim, (batch, 1)
            ),
            x.shape,
        )
        x = (x - mean) / ops.power_scalar(var + self.eps, 0.5) * ops.broadcast_to(
            ops.reshape(self.weight, (1, self.dim)), x.shape
        ) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = init.randb(
                *x.shape, p=(1 - self.p), dtype="float32", device=x.device
            )
            x = ops.mul_scalar(ops.multiply(x, mask), 1 / (1 - self.p))
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return ops.add(x, self.fn(x))
