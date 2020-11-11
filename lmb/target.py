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

    def correct(self):
        """
        Correct the target state based on the new measurement
        """
        pass

    def __repr__(self):
        """
        String representation of object
        """
        return "T({} / {}: {})".format(self.label, self.log_r, self.pdf.mc)

