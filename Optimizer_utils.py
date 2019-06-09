import matplotlib.pyplot as plt
from functions_utils import *
from armijo_utils import Armijo_method
from newton_utils import NewtonMethod
from penalty_method import AugmentedLagrangian


class Gradient_descent():
    def __init__(self, method_type='steepest_descent', threshold=0.00001, step_size_estimator=Armijo_method(),
                 max_steps=100000, verbose=True):
        self.threshold = threshold
        self.max_steps = max_steps
        self.verbose = verbose
        self.f_val_list = []
        self.step_sizes_list = []
        self.step_size_estimator = step_size_estimator
        self.method_type = method_type

    def optimize(self, func, start_point):
        self.f_val_list.append(func.val(start_point))

        x = start_point

        for step in range(self.max_steps):
            print("step:", step)
            prev_x = x
            if self.method_type == 'steepest_descent':
                x = self.optimizer_step(x, func)
            elif self.method_type == 'newton_method':
                x = self.optimizer_step_newton(x, func)
            else:
                print("Direction method not selected")
                break
            self.f_val_list.append(func.val(x))

            # if self.verbose:
            #     print("f(x)=", func.val(x), " current point= ~", np.round(x, 5))

            # print("norm=",np.linalg.norm(func.grad(x)))
            if np.linalg.norm(func.grad(x)) < self.threshold:
                print("Optimizer reached accuracy threshold after", step, "iterations!")
                break
        return x

    def optimizer_step(self, x, func):
        step_size = self.step_size_estimator.calc_step_size(x, func, direction=func.grad(x))
        x = x - step_size * func.grad(x)
        # self.step_size_estimator.armijo_plot()
        self.step_sizes_list.append(step_size)
        return x

    def optimizer_step_newton(self, x, func):
        newton = NewtonMethod()
        d = newton.direction(x, func)
        step_size = self.step_size_estimator.calc_step_size(x, func, direction=d)
        x = x - step_size * d
        self.step_sizes_list.append(step_size)
        return x

    def plot_step_sizes(self):
        iterations_list = range(len(self.step_sizes_list))

        a, = plt.plot(iterations_list, self.step_sizes_list, label='step size')
        plt.legend(handles=[a])
        plt.ylabel('step size')
        plt.xlabel('iterations')
        plt.show()

    def get_convergence(self, val_optimal):
        '''
        gets converg rates list
        :param f_list: list of values of f during gradient descent algo
        :param val_optimal: the global minimum value of the function
        '''
        converg_list = []
        iterations_list = []
        for idx, val in enumerate(self.f_val_list):
            converg_list.append(val - val_optimal)
            iterations_list.append(idx)

        return iterations_list, converg_list

    def plot_convergence(self, val_optimal, f_name='plot title', marker=None, save = True):
        '''
        plots the convergence rate
        :param f_list: list of values of f during gradient descent algo
        :param val_optimal: the global minimum value of the function
        '''
        converg_list = []
        iterations_list = []
        for idx, val in enumerate(self.f_val_list):
            #converg_list.append(abs(val - val_optimal))
            converg_list.append(val)
            iterations_list.append(idx)

        plt.plot(iterations_list, converg_list)
        plt.ylabel('f(x)-f* / log')
        plt.xlabel('iterations')
        #plt.yscale('log')
        label = f_name + ' - ' + self.method_type + ' convergence rate'
        plt.title(label)
        if marker != None:
            x, y = marker
            plt.plot(x, y, 'ro')
        plt.gcf()
        name = label + '_fig.JPEG'
        plt.savefig(name, bbox_inches='tight')
        plt.show()

