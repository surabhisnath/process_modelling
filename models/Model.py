class Model:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.features = features
        self.sim_mat = self.get_similarity_matrix()
        self.freq = self.get_frequencies()
        self.unique_responses = list(self.freq.keys())

    def fit():
        pass
    def simulate():
        pass
    def test():
        pass