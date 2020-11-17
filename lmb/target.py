import numpy as np

class Target():
    """
    Represents a single target

    A target is represented by its label, an existence probability, and 
    a probability density function describing the current target state.
    """
    def __init__(self, label, log_r=0.0, pdf=None):
        self.label = label
        self.log_r = log_r
        self.pdf = pdf

    def predict(self, log_p_survival):
        """
        Predict the state to the next time step

        Parameters
        ----------
        log_p_survival: log of survival probability
        """
        # Predict PDF
        self.pdf.predict()
        # Predict existence probability r
        self.log_r += log_p_survival

    def correct(self, assignment_weights):
        """
        Correct the target state based on the computed measurement associations

        Updates the existence probability and combines the association PDFs
        into the resulting PDF. 

        Parameters
        ----------
        assignment_weights : array_like (len(assignments + 1))
            Computed hypothesis weights from ranked assignment (e.g. Gibbs sampler)
        """
        # 1.: self.log_r = sum of assignment weights
        # 2.: Combine PDFs

    def create_associations(self, z):
        """
        Compute new hypothetical target-measurement associations

        TODO: define structure and storage of associations

        Parameters
        ----------
        z : array_like
            measurements
        
        Returns
        -------
        out : array_like (len(z))
            computed etas (weights) of all target-measurement associations
        """
        # TODO: Either create copies of PDFs and subsequentially call correct(z)
        # or handle all associations in one matrix using broadcasting
        # 1. Compute PDFs and weights for each association and a missed detection
        # 2. Calculate etas by adding self.log_r
        pass

    def nll_false(self):
        """
        Negative log-likelihood of target being false (died or not born)
        """
        return -np.log(1 - np.exp(self.log_r))

    def __repr__(self):
        """
        String representation of object
        """
        return "T({} / {}: {})".format(self.label, self.log_r, self.pdf.mc)

