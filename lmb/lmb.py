class LMB():
    """
    Main class of the labeled multi bernoulli filter implementation.
    """
    def __init__(self):
        pass

    def update(self,z):
        """
        Main function to update the internal tracker state with a measuerement

        Parameters
        ----------
        z: measurement object (class to be implemented)

        Returns
        -------
        out: array_like
            updated tracks
        """
        self.predict()
        self.correct(z)

        return self.select()

    def predict(self):
        """
        Prediction step of LMB tracker

        Predicts tracker states and existence probabilities
        """
        pass

    def correct(self, z):
        """
        Correction step of LMB tracker

        Correct the predicted track states and their existence probabililities using the new measurement

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        self._prune()

    def _prune(self):
        """
        Pruning of tracks

        Selection according to the configured pruning threshold for the existence probability.
        Afterwards, limit the remaining tracks to the configured maximum number based on 
        descending existence probability. 
        """
        pass

    def select(self):
        """
        Select tracks based on existence probability

        Computes the most likely number of tracks and selects for this number of tracks the Gaussian
        mixture component with the highest weight of the tracks with the highest existence probability.
        """
        pass
