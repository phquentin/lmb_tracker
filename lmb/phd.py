class PHD():
    """
    Implementation of the PHD using Gaussian mixtures
    """
    
    def __init__(self):
        pass

    def _predict_phd(self):
        """
        Predict Gaussian Mixture state components (m, P) of all tracks
        """
        pass

    def _correct_phd(self, z):
        """
        Update PHD with new measurements and calculate log-likelihood of track-measurement associations

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        pass

    def _merge_phd(self):
        """
        Merge Gaussian Mixture components which are closer than a defined threshold
        """
        pass