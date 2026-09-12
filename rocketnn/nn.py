"""A from-scratch multilayer perceptron.

Deliberately tiny and transparent: no PyTorch/TensorFlow. Every weight is a
plain NumPy array, the forward pass is a handful of matmuls, and the backward
pass is written out by hand. This matters because the same arithmetic is
re-implemented in C++ (`firmware/nn_inference.h`) for the Teensy, and we verify
the two produce identical numbers. Owning the math on both sides is what makes
that guarantee possible.

Architecture: fully-connected layers with ReLU on the hidden layers and a
linear output layer. ReLU is chosen on purpose - it is a single `max(0, x)` on
the microcontroller, with no `exp`/`tanh` table lookups in the hot path.
"""

from __future__ import annotations

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


class MLP:
    """Multilayer perceptron with ReLU hidden layers and a linear output.

    Parameters
    ----------
    sizes : list[int]
        Layer widths, e.g. ``[10, 24, 16, 6]`` = 10 inputs, two hidden layers
        of 24 and 16 units, and 6 outputs.
    seed : int
        RNG seed for reproducible weight initialisation.
    """

    def __init__(self, sizes, seed=0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        # He initialisation, appropriate for ReLU networks.
        for nin, nout in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / nin)
            self.W.append(rng.standard_normal((nout, nin)) * scale)
            self.b.append(np.zeros(nout))
        self._init_adam()

    # -- forward / backward -------------------------------------------------
    def forward(self, X, cache=None):
        """Forward pass for a batch ``X`` of shape (N, n_in).

        Returns the output of shape (N, n_out). If a dict is passed as
        ``cache``, intermediate activations are stored for ``backward``.
        """
        A = X
        if cache is not None:
            cache["A"] = [A]
            cache["Z"] = []
        n_layers = len(self.W)
        for i in range(n_layers):
            Z = A @ self.W[i].T + self.b[i]
            if i < n_layers - 1:
                A = relu(Z)
            else:
                A = Z  # linear output layer
            if cache is not None:
                cache["Z"].append(Z)
                cache["A"].append(A)
        return A

    def backward(self, dOut, cache):
        """Backprop. ``dOut`` is dLoss/dOutput of shape (N, n_out).

        Returns (dW, db) lists matching ``self.W`` / ``self.b``.
        """
        A = cache["A"]
        Z = cache["Z"]
        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers
        dA = dOut
        for i in reversed(range(n_layers)):
            if i < n_layers - 1:
                dZ = dA * (Z[i] > 0.0)  # ReLU derivative
            else:
                dZ = dA  # linear output
            dW[i] = dZ.T @ A[i]
            db[i] = dZ.sum(axis=0)
            dA = dZ @ self.W[i]
        return dW, db

    # -- Adam optimiser -----------------------------------------------------
    def _init_adam(self):
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def adam_step(self, dW, db, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i in range(len(self.W)):
            self._mW[i] = beta1 * self._mW[i] + (1 - beta1) * dW[i]
            self._vW[i] = beta2 * self._vW[i] + (1 - beta2) * (dW[i] ** 2)
            self.W[i] -= lr * (self._mW[i] / bc1) / (np.sqrt(self._vW[i] / bc2) + eps)

            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * db[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * (db[i] ** 2)
            self.b[i] -= lr * (self._mb[i] / bc1) / (np.sqrt(self._vb[i] / bc2) + eps)

    # -- persistence --------------------------------------------------------
    def save(self, path, extra=None):
        data = {}
        for i in range(len(self.W)):
            data[f"W{i}"] = self.W[i].astype(np.float64)
            data[f"b{i}"] = self.b[i].astype(np.float64)
        data["sizes"] = np.array(self.sizes)
        if extra:
            for k, v in extra.items():
                data[k] = np.asarray(v)
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        sizes = list(int(x) for x in d["sizes"])
        net = cls(sizes)
        for i in range(len(sizes) - 1):
            net.W[i] = d[f"W{i}"]
            net.b[i] = d[f"b{i}"]
        net._init_adam()
        return net, d

    def forward_f32(self, x):
        """Single-sample forward pass done entirely in float32.

        This mirrors the Teensy's arithmetic (32-bit floats) and is what the
        C++ parity test is compared against, so the numbers line up.
        """
        a = np.asarray(x, dtype=np.float32)
        n_layers = len(self.W)
        for i in range(n_layers):
            W = self.W[i].astype(np.float32)
            b = self.b[i].astype(np.float32)
            z = (W @ a + b).astype(np.float32)
            a = np.maximum(z, np.float32(0.0)) if i < n_layers - 1 else z
        return a
