class TripySequentialEncoderWithLearnedSimilarityModule:
    def __init__(self, ndp_module):
        self._ndp_module = ndp_module

    def debug_str(self):
        raise NotImplementedError()

    def similarity_fn(self, query_embeddings, item_ids, item_embeddings=None, **kwargs):
        assert len(query_embeddings.shape) == 2, "len(query_embeddings.shape) must be 2"
        assert len(item_ids.shape) == 2, "len(item_ids.shape) must be 2"
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)
        assert len(item_embeddings.shape) == 3, "len(item_embeddings.shape) must be 3"
        return self._ndp_module(
            query_embeddings=query_embeddings,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            **kwargs,
        )
