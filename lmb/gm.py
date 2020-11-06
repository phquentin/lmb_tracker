class GM():
    """
    Implementation of the PHD using Gaussian mixtures
    """
    
    def __init__(self):
        pass

    def predict(self):
        """
        Predict Gaussian Mixture state components (m, P) of all tracks
        """
        pass

    def correct(self, z):
        """
        Update PHD with new measurements and calculate log-likelihood of track-measurement associations

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        pass

    def merge(self):
        """
        Merge Gaussian Mixture components which are closer than a defined threshold
        """
        pass