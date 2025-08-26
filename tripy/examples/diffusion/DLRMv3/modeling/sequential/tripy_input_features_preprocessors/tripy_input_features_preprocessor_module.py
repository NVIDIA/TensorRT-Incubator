import abc

class TripyInputFeaturesPreprocessorModule:
    @abc.abstractmethod
    def debug_str(self):
        pass

    @abc.abstractmethod
    def forward(self, past_lengths, past_ids, past_embeddings, past_payloads):
        pass
