# Tripy version of RelativeAttentionBiasModule and RelativePositionalBias
import abc
import nvtripy as tp
import numpy as np

class RelativeAttentionBiasModule(tp.Module):
    @abc.abstractmethod
    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            float tensor broadcastable to [B, N, N]
        """
        pass

class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        w_init = np.random.normal(loc=0, scale=0.02, size=(2 * max_seq_len - 1)).astype(np.float32)
        self._w = tp.Tensor(w_init)

    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        # all_timestamps is ignored
        n = self._max_seq_len
        # Pad self._w to length 2*n-1 + n zeros at the end
        w_padded = tp.pad(self._w[:2*n-1], [(0, n)])
        # Repeat the whole padded vector n times (concatenate n copies)
        t = tp.concatenate([w_padded] * n, dim=0)
        t = t[:-n]
        t = tp.reshape(t, (1, n, 3*n-2))
        r = (2*n-1)//2
        return t[..., r:-r]
