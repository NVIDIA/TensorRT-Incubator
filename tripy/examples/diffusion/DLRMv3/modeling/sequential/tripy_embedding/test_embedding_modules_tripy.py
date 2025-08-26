
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import pytest
import nvtripy as tp
from sequential.embedding_modules import LocalEmbeddingModule, CategoricalEmbeddingModule
from sequential.tripy_embedding.embedding_modules_tripy import TripyLocalEmbeddingModule, TripyCategoricalEmbeddingModule

def torch_to_tripy_tensor(tensor):
    return tp.Tensor(tensor.detach().cpu().numpy().astype(np.int32 if tensor.dtype in [torch.int32, torch.int64] else np.float32))

def test_local_embedding_equivalence():
    num_items = 10
    emb_dim = 4
    torch_mod = LocalEmbeddingModule(num_items, emb_dim)
    tripy_mod = TripyLocalEmbeddingModule(num_items, emb_dim)
    # Copy weights
    tripy_mod._item_emb.weight = tp.Tensor(torch_mod._item_emb.weight.detach().cpu().numpy())
    # Test for random item_ids
    item_ids = torch.randint(0, num_items+1, (5,))
    torch_out = torch_mod.get_item_embeddings(item_ids)
    tripy_out = tripy_mod.get_item_embeddings(torch_to_tripy_tensor(item_ids))
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), tripy_out.tolist(), rtol=1e-5, atol=1e-6)

def test_categorical_embedding_equivalence():
    num_items = 10
    emb_dim = 4
    # Each item maps to a category (simulate 5 categories)
    item_id_to_category_id = torch.randint(0, 5, (num_items,))
    torch_mod = CategoricalEmbeddingModule(num_items, emb_dim, item_id_to_category_id)
    tripy_mod = TripyCategoricalEmbeddingModule(num_items, emb_dim, tp.Tensor(item_id_to_category_id.numpy()))
    # Copy weights
    tripy_mod._item_emb.weight = tp.Tensor(torch_mod._item_emb.weight.detach().cpu().numpy())
    # Test for random item_ids
    item_ids = torch.randint(1, num_items+1, (5,))
    torch_out = torch_mod.get_item_embeddings(item_ids)
    tripy_out = tripy_mod.get_item_embeddings(torch_to_tripy_tensor(item_ids))
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), tripy_out.tolist(), rtol=1e-5, atol=1e-6)
