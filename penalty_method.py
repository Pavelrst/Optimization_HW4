import numpy as np


class AugmentedLagrangian:
    def __init__(self, objective, constraints, start_point, optimal, name):
        self.f = objective
        self.constr_list = constraints
        self.mu_list = [1]*len(constraints)
        self.start = start_point
        self.name = name
        self.optimal = optimal

        self.p_max = 1000
        self.curr_p = 2
        self.p_alpha = 10
        self.penalty = Penalty(2)

    def print_multipliers(self):
        print("Augmented Lagrangian multipliers are: ", self.mu_list)

    def update_mu(self, x):
        '''
        Updates Lagrangian multipliers
        '''
        for idx, mu in enumerate(self.mu_list):
            mu_new = self.mu_list[idx] * self.penalty.deriv(self.constr_list[idx].val(x))
            self.mu_list[idx] = mu_new
        #print("self.mu_list=", self.mu_list)

    def update_p(self):
        if self.curr_p < self.p_max:
            self.curr_p = self.curr_p * self.p_alpha
            self.penalty = Penalty(min(self.curr_p, self.p_max))

    def optimal(self):
        return self.optimal()

    def name(self):
        return self.name()

    def starting_point(self):
        return self.start

    def val(self, x):
        res = self.f.val(x)
        for mu, constr in zip(self.mu_list, self.constr_list):
            res += mu * self.penalty.val(constr.val(x))
        return res

    def grad(self, x):
        res = self.f.grad(x)
        for mu, constr in zip(self.mu_list, self.constr_list):
            res += mu * self.penalty.deriv(constr.val(x))*constr.grad(x)
        return res

    def hess(self, x):
        res = self.f.hess(x)
        for mu, constr in zip(self.mu_list, self.constr_list):
            temp = np.matmul(constr.grad(x), np.transpose(constr.grad(x)))
            res += mu * self.penalty.sec_deriv(constr.val(x))*temp
        return res


class Penalty:
    def __init__(self, p=2):
        self.p = p

    def val(self, g):
        if self.p*g >= -0.5:
            return (1 / self.p) * (0.5 * ((self.p*g) ** 2) + self.p*g)

        else:
            return (1 / self.p) * (-0.25 * (np.log(-2 * self.p*g) - 3 / 8))

    def deriv(self, g):
        if self.p*g >= -0.5:
            return self.p*g + 1
        else:
            return - (1 / (4*self.p*g))


    def sec_deriv(self, g):
        if self.p*g >= -0.5:
            return self.p
        else:
            return 1 / (4 * self.p * g**2)
