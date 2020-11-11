import numpy as np

class GM():
    """
    Gaussian Mixture PDF

    Parameters
    ----------
    x0 : numpy.array(dim_x) optional
        Initial state estimate
    dim_x : int
        Dimension (number) of state variables
    dim_z : int
        Dimension (number) of measurement inputs
    P_init : numpy.array(dim_x, dim_x)
        Initial state covariance matrix
    F : numpy.array(dim_x, dim_x)
        State Transition matrix
    Q : numpy.array(dim_x, dim_x)
        Process Noise matrix
    H : numpy.array(dim_z, dim_x)
        Observation model
    R : numpy.array(dim_z, dim_z)
        Observation Noise matrix
    """
    
    def __init__(self, params, x0=None):
        self.params = params
        self.dim_x = self.params.dim_x
        self.F = self.params.F
        self.H = self.params.H
        self.Q = self.params.Q
        self.R = self.params.R
        # data type of mixture component entry
        self.dtype_mc = np.dtype([('log_w', 'f8'),
                                ('x', 'f4', self.dim_x),
                                ('P', 'f4', (self.dim_x, self.dim_x))])
        # init list of mixture components with one Gaussian
        self.mc = np.zeros(1, dtype=self.dtype_mc)
        self.mc[0]['log_w'] = 1.0
        self.mc[0]['x'] = x0 if x0 is not None else np.zeros(self.dim_x)
        self.mc[0]['P'] = self.params.P_init

    def predict(self):
        """
        Predict Gaussian Mixture state components (x, P) 
        
        Each mixture component is predicted using the Kalman equations
            x = Fx
            P = FPF' + Q
        """
        # x and P of all mixture components are predicted with array broadcasting
        # In their original form, shapes of x (mc['x']) and F are not aligned,
        # thus x has to be transposed.
        # F: (dim_x, dim_x)
        # x: (len(mc), dim_x)
        # -> x = (Fx')' -> x = xF'
        self.mc['x'] = np.dot(self.mc['x'], self.F.T)
        # Iterate through all mixture components
        for i in range(len(self.mc)):
            self.mc[i]['P'] = np.dot(np.dot(self.F, self.mc[i]['P']), self.F.T)
        # Add Process Noise matrix via broadcasting
        self.mc['P'] = self.mc['P'] + self.Q

    def correct(self, z):
        """
        Update PDF with a new measurement

        Corrects each mixture component and their weights 
        with the new measurement using the Kalman filter
            x = x + Ky
            P = P - KSK'
            with
                y = z - Hx
                S = HPH' + R
                K = PH'inv(S)

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        ## for each mixture component or via broadcasting:
        #   y = z - np.dot(self.H, self.mc['x'])
        #   S = np.dot(np.dot(self.H, self.mc['P']), self.H.T) + self.R
        #   S_inv = np.linalg.inv(S)
        #   K = np.dot(np.dot(self.mc['P'], self.H.T), S_inv)
        #   self.mc['x'] = self.mc['x'] + np.dot(K, y)
        #   self.mc['P'] = self.mc['P'] - np.dot(np.dot(K, S), K.T)
        #   self.mc['log_w'] = self.mc['log_w'] + 2 * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + np.dot(y, np.dot(S_inv, y))
        #   self.mc['log_w'] *= -0.5
        #
        ## Normalization of mixture weights
        # log_w_sum = logsumexp(self.mc['log_w'])
        # self.mc['log_w'] -= log_w_sum
        pass # @todo Implementation

    def merge(self):
        """
        Merge Gaussian Mixture components which are closer than a defined threshold
        """
        pass
