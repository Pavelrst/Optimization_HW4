import numpy as np

class NewtonMethod:
    def direction(self, x, func):
        try:
            H_inv = np.linalg.inv(func.hess(x))
        except:
            H_inv = np.linalg.pinv(func.hess(x))
        return np.matmul(H_inv, func.grad(x))

