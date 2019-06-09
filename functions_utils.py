from scipy.optimize import rosen
import numpy as np
from numdiff_utils import numdiff
from scipy.io import loadmat

class Quadratic_general:
    def __init__(self, Q, d, e):
        '''
        xT Q x + dT x + e
        '''
        self.Q = Q
        self.d = d
        self.e = e

    def val(self, x):
        xT = np.transpose(x)
        xT_Q_x = np.matmul(np.matmul(xT,self.Q),x)
        dT_x = np.matmul(np.transpose(self.d),x)
        return (1/2 * xT_Q_x + dT_x + self.e)[0]

    def grad(self, x):
        Q_QT = self.Q+np.transpose(self.Q)
        return (1/2 * np.matmul(Q_QT,x) + self.d)

    def hess(self, x):
        return 1/2 * self.Q+np.transpose(self.Q)


