from functions_utils import Quadratic_general
from Optimizer_utils import Gradient_descent
from penalty_method import AugmentedLagrangian
import matplotlib.pyplot as plt
import numpy as np

def main():
    # opt_grad = Gradient_descent(method_type='steepest_descent', max_steps=50, verbose=False)
    objective, constraints_list, start_point, optimal, name = create_problem()

    f = AugmentedLagrangian(objective, constraints_list, start_point, optimal, name)

    #opt_grad.optimize(f, f.starting_point())
    #opt_grad.plot_convergence(f.optimal, f.name)

    opt_newton = Gradient_descent(method_type='newton_method', max_steps=1000, verbose=True)

    x = f.starting_point()
    for i in range(7):
        print("x=", x)
        x = opt_newton.optimize(f, x)
        f.update_mu(x)
        f.update_p()

    print("The solution is:\n", x)
    print("Lagrange multipliers are:\n", f.get_mu())

    # opt_newton.plot_convergence(f.optimal, f.name)


def plot_problem(objective, constraints_list):
    x_list = np.arange(-10,10,0.1)
    for constr in constraints_list:
        c_val_list = []
        for x in x_list:
            c_val_list.append(constr.val(x))
        plt. plot(x_list,c_val_list)
    plt.show()


def create_problem():
    start_point = np.array([[1],
                            [1]])
    optimal = 37+2/3
    name = 'Task 2'
    # objective
    Q = np.array([[4, 0],
                 [0, 2]])
    b = np.array([[-20],
                 [-2]])
    e = 51
    objective = Quadratic_general(Q, b, e)

    # Constarints
    constraints_list = []

    # 1
    Q_zeros = np.array([[0, 0],
                        [0, 0]])
    b = np.array([[0.5],
                  [1]])
    e = -1
    constraints_list.append(Quadratic_general(Q_zeros, b, e))

    # 2
    b = np.array([[1],
                  [-1]])
    e = 0
    constraints_list.append(Quadratic_general(Q_zeros, b, e))

    # 3
    b = np.array([[-1],
                  [-1]])
    e = 0
    constraints_list.append(Quadratic_general(Q_zeros, b, e))

    return objective, constraints_list, start_point, optimal, name

if __name__ == "__main__":
    main()

