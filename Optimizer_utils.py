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
        self.augmented_lagrangian_grad_list = []
        self.distance_to_optimal_point_list = []
        self.distance_to_optimal_multipliers_list = []
        self.most_violated_constraints_list = []
        self.step_size_estimator = step_size_estimator
        self.method_type = method_type
        self.global_step = 0

    def optimize(self, func, start_point):
        #self.f_val_list.append(func.val(start_point))

        x = start_point

        for step in range(self.max_steps):
            print("step:", self.global_step)
            self.global_step += 1
            prev_x = x
            if self.method_type == 'steepest_descent':
                x = self.optimizer_step(x, func)
            elif self.method_type == 'newton_method':
                x = self.optimizer_step_newton(x, func)
            else:
                print("Direction method not selected")
                break

            # Adding values for plotting
            self.f_val_list.append(func.val(x))
            self.augmented_lagrangian_grad_list.append(np.linalg.norm(func.grad(x)))
            self.distance_to_optimal_point_list.append(np.linalg.norm(func.optimal_x - x))

            opt_mu = np.array(func.optimal_mu)
            curr_mu = np.array(np.array(func.get_mu()))
            dist = np.linalg.norm(curr_mu - opt_mu)
            self.distance_to_optimal_multipliers_list.append(dist)

            most = func.get_most_violated_constraint(x)
            self.most_violated_constraints_list.append(most)

            #print("f(x)=", func.val(x), " current point= ~", np.round(x, 5))

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
            converg_list.append(abs(val - val_optimal))
            #converg_list.append(val)
            iterations_list.append(idx)

        # Convergence
        plt.plot(iterations_list, converg_list, label='convergence')
        plt.ylabel('f(x)-f* / log')
        plt.xlabel('Newton iterations')
        plt.yscale('log')
        label = 'Convergence_rate'
        plt.title(label)
        plt.legend()
        plt.gcf()
        name = label + '_fig.JPEG'
        plt.savefig(name, bbox_inches='tight')
        plt.show()

        # Gradient
        plt.plot(iterations_list, self.augmented_lagrangian_grad_list, label='augmented_lagrangian_grad')
        plt.ylabel('|grad_f(x)| / log')
        plt.xlabel('Newton iterations')
        plt.yscale('log')
        label = 'Augmented_Lagrangian_gradient'
        plt.title(label)
        plt.legend()
        plt.gcf()
        name = label + '_fig.JPEG'
        plt.savefig(name, bbox_inches='tight')
        plt.show()

        # multipliers and point
        plt.plot(iterations_list, self.distance_to_optimal_multipliers_list, label='distance_to_optimal_multipliers')
        plt.plot(iterations_list, self.distance_to_optimal_point_list, label='distance_to_optimal_point')
        plt.ylabel('Distance / log')
        plt.xlabel('Newton iterations')
        plt.yscale('log')
        label = 'Optimal_point_and_optimal_multipliers_Distance'
        plt.title(label)
        plt.legend()
        plt.gcf()
        name = label + '_fig.JPEG'
        plt.savefig(name, bbox_inches='tight')
        plt.show()

        # most violated
        plt.plot(iterations_list, self.most_violated_constraints_list, label='most violated constraint')
        plt.ylabel('Largest violation / log')
        plt.xlabel('Newton iterations')
        plt.yscale('log')
        label = 'Maximal_constraint_violation'
        plt.title(label)
        plt.legend()
        plt.gcf()
        name = label + '_fig.JPEG'
        plt.savefig(name, bbox_inches='tight')
        plt.show()
