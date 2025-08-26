# Tripy version of RelativeBucketedTimeAndPositionBasedBias
import nvtripy as tp
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sequential.tripy_hstu.relative_attention_bias_module import RelativeAttentionBiasModule

class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """
    def __init__(self, max_seq_len: int, num_buckets: int, bucketization_fn):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._num_buckets = num_buckets
        # Initialize weights as in torch version
        ts_w_init = np.random.normal(loc=0, scale=0.02, size=(num_buckets + 1)).astype(np.float32)
        pos_w_init = np.random.normal(loc=0, scale=0.02, size=(2 * max_seq_len - 1)).astype(np.float32)
        self._ts_w = tp.Tensor(ts_w_init)
        self._pos_w = tp.Tensor(pos_w_init)
        self._bucketization_fn = bucketization_fn

    def forward(self, all_timestamps: tp.Tensor) -> tp.Tensor:
        B = all_timestamps.shape[0]
        N = self._max_seq_len
        # Pad pos_w to length 2*N-1 + N zeros at the end
        t = tp.pad(self._pos_w[:2*N-1], [(0, N)])
        t = tp.concatenate([t] * N, dim=0)
        t = t[:-N]
        t = tp.reshape(t, (1, N, 3*N-2))
        r = (2*N-1)//2
        rel_pos_bias = t[..., r:-r]
        # [B, N+1] for easier manipulation
        ext_timestamps = tp.concatenate([all_timestamps, all_timestamps[:, N-1:N]], dim=1)
        # bucketed_timestamps: [B, N, N]
        bucketed = self._bucketization_fn(
            tp.unsqueeze(ext_timestamps[:, 1:], 2) - tp.unsqueeze(ext_timestamps[:, :-1], 1)
        )
        bucketed_timestamps = tp.minimum(tp.maximum(bucketed, 0), self._num_buckets)
        # rel_ts_bias: [B, N, N]
        rel_ts_bias = tp.gather(self._ts_w, 0, bucketed_timestamps.reshape((-1,))).reshape((B, N, N))
        return rel_pos_bias + rel_ts_bias
