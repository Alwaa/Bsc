import scipy
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
import numpy as np



def COBYLA(problem):
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Function (z)"]
    con = lambda x: constraintf[0](x[None])
    fun = lambda x: costf(x[None])
    #con = lambda x: constraintf[0](x)
    #fun = lambda x: costf(x)
    nlc = NonlinearConstraint(con, -1, 0)

    bounds = problem["Bounds"]
    #x = np.array([1,2])
    #A = np.array([1,0])
    #B = np.array([0,1])
    #print(A.dot(x),B.dot(x))
    #print(bounds)
    #LinearConstraint(A,bounds[0], bounds[1], keep_feasible=True)} # Not for cobula??

    constraints = [{"type": "ineq","fun": lambda x: x[0] + bounds[0]},
                         {"type": "ineq","fun": lambda x: -x[0] + bounds[1]},
                         {"type": "ineq","fun": lambda x: x[1] + bounds[0]},
                         {"type": "ineq","fun": lambda x: -x[1] + bounds[1]},
                         {"type": "ineq","fun": lambda x: 0.1-con(x)[0]}]

    for i in range(10):
        rnd = np.random.RandomState(42 + i)
        x = ((rnd.rand(2)*(bounds[1]-bounds[0])))+bounds[0]
        print("-----\n",x)

        opt = scipy.optimize.minimize(fun, x, constraints = constraints, method="COBYLA") #,tol = 0.1 #catol does not work???
        #Also a lot of infeasable solutions in eikson paper
        x = opt.x
        print(opt.message)
        print(x,fun(x))
        print(con(x)[0])
