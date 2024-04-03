class HoldoutSplit:
    def __init__(self, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        from sklearn.model_selection import train_test_split
        indices = range(len(X))
        train_indices, test_indices = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)
        yield train_indices, test_indices
