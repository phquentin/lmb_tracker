class Target():
    """
    Represents a single target

    A target is represented by its label, an existence probability, and 
    a probability density function describing the current target state.
    """
    def __init__(self, label, log_r=1.0, pdf=None):
        self.label = label
        self.log_r = log_r
        self.pdf = pdf

    def predict(self):
        """
        Predict the state to the next time step
        """
        # Predict PDF
        self.pdf.predict()
        # @todo Predict existence probability r

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

