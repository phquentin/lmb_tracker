from copy import deepcopy
import numpy as np

from scipy.special import logsumexp

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
        assignment_weights : array_like (len(self.assignments))
            Computed hypothesis weights (in log-likelihood) from ranked assignment
        """
        # 1.: self.log_r = sum of assignment weights
        self.log_r = logsumexp(assignment_weights)
        print('log_r=',self.log_r, ' r=', np.exp(self.log_r))
        # 2.: Combine PDFs
        self.pdf.overwrite_with_merged_pdf(self.assignments, assignment_weights - self.log_r)

    def create_assignments(self, Z):
        """
        Compute new hypothetical target-measurement associations

        Parameters
        ----------
        Z : array_like
            measurements
        
        Returns
        -------
        out : array_like (len(z))
            Negative log-likelihood of computed etas (weights) of all target-measurement associations
        """
        # Compute PDFs and weights for each association and a missed detection
        self.assignments = []
        for z in Z:
            self.assignments.append(deepcopy(self.pdf).correct(z['z']))

        self.assignments.append(deepcopy(self.pdf).correct(None))
        # Calculate etas by adding self.log_r
        nll_etas = [- (self.log_r + pdf.log_eta_z) for pdf in self.assignments]
        return nll_etas

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

