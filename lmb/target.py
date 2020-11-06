class Target():
    """
    Represents a single target

    A target is represented by its label, an existence probability, and 
    a probability density function describing the current target state.
    """
    def __init__(self, label, r=0, pdf=None):
        self.label = label
        self.r = r
        self.pdf = pdf

    def predict(self):
        """
        Predict the state to the next time step
        """
        pass

    def correct(self):
        """
        Correct the target state based on the new measurement
        """
        pass

