import nvtripy as tp

def tripy_get_current_embeddings(lengths, encoded_embeddings):
    """
    Args:
        lengths: (B,) x int
        encoded_embeddings: (B, N, D,) x float
    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.shape
    flattened_offsets = (lengths - tp.Tensor(1)) + tp.arange(B, dtype=lengths.dtype) * N
    flat = tp.reshape(encoded_embeddings, (B * N, D))
    selected = tp.gather(flat, 0, flattened_offsets)
    return tp.reshape(selected, (B, D))
