import numpy as np
import matplotlib.pyplot as plt


class Armijo_method():
    def __init__(self, initial_alpha=1, sigma=0.25, beta=0.5):
        self.initial_alpha = initial_alpha
        self.sigma = sigma
        self.beta = beta

    def calc_step_size(self, x, func, direction):
        '''
        Denotre Phi(alpha) as function of step size: Phi(alpha)=f(x+alpha*d)-f(x)
        while d is direction. Deriviate Phi by alpha and we get:
        Phi'(alpha) =  f'(x+alpha*d)*d so,
        Phi'(0) =  f'(x)*d i.e. directional derivative, in our case - gradient
        Denote: c = Phi'(0) = f_grad(x), so
        alpha*c is a tangent line to Phi(alpha) at alpha=0.
        Now lets make a new line: sigma*alpha*c
        Finally let's state to Armijo condition:
            f(x+alpha*d)-f(x) <= sigma*alpha*c
        Also we assume search direction d is gradient direction.
        :param x: the point from which we want make a step
        :param alpha: Initial step size
        :param sigma: Factor for a slope of tangent
        :param beta: Step size decrement factor
        :return: step size which is fulfilling Armijo condition.
        '''

        self.armijo = ArmijoPhiFunc(x, func.val, func.grad,
                                    direction=direction, sigma=0.25, beta=0.5)

        alpha = self.initial_alpha
        # f(x+alpha*d)-f(x) <= sigma*alpha*c
        while self.armijo.phi_val(alpha) > self.armijo.elevated_tangent_val(alpha):
            alpha = alpha * self.beta
        return alpha

    def armijo_plot(self):
        alpas = np.linspace(0, 1, num=1000)
        phi_list = []
        tangent_list = []
        elevated_tangent_list = []
        for idx, alpha in enumerate(alpas):
            phi_list.append(self.armijo.phi_val(alpha))
            tangent_list.append(self.armijo.tanget_val(alpha))
            elevated_tangent_list.append(self.armijo.elevated_tangent_val(alpha))

        a, = plt.plot(alpas, phi_list, label='phi')
        b, = plt.plot(alpas, tangent_list, label='tangent')
        c, = plt.plot(alpas, elevated_tangent_list, label='elevated_tangent')
        plt.legend(handles=[a, b, c])
        plt.ylabel('Phi(alpha)')
        plt.xlabel('alpha')
        plt.grid()
        plt.show()


class ArmijoPhiFunc:
    def __init__(self, x, func_val, func_grad, direction, sigma=0.25, beta=0.5):
        self.f = func_val
        self.g = func_grad
        self.x = x
        self.direction = direction
        self.sigma = sigma
        self.beta = beta

    def phi_val(self, alpha):
        # The direction is opposite to gradient.
        alpha_d = -alpha*self.direction
        val = self.f(self.x + alpha_d) - self.f(self.x)
        return val

    def tanget_val(self, alpha):
        '''
        :param alpha: given step size.
        :return: value of tangent to phi at given alpha.
        '''
        c = -np.matmul(self.g(self.x), self.direction)
        #c = -np.matmul(self.direction, self.direction)
        return alpha * c

    def elevated_tangent_val(self, alpha):
        '''
        :param alpha: given step size.
        :return: value of tangent to phi at given alpha.
        '''
        c = -np.matmul(np.transpose(self.g(self.x)), self.direction)
        #c = np.matmul(np.transpose(self.g(self.x)), self.direction)
        return self.sigma * alpha * c



