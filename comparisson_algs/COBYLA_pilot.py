import scipy
#from scipy.optimize import NonlinearConstraint
#from scipy.optimize import LinearConstraint
import scipy.optimize as opt
import numpy as np

def scipy_bounds(problem_bounds):
    lbs = problem_bounds[0::2]
    ubs = problem_bounds[1::2]
    assert len(lbs) == len(ubs), "Bounds not defined correctly in scipy bounds object"
    keep_feas = [True]*len(ubs)
    bounds = opt.Bounds(lbs,ubs,keep_feasible=keep_feas)

    return bounds

def cobyla_run(problem, x0, obj_tol = 0.1, maxiter = 100):
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    con = lambda x: constraintf[0](x[None])
    fun = lambda x: costf(x[None])
    #con = lambda x: constraintf[0](x)
    #fun = lambda x: costf(x)
    #nlc = NonlinearConstraint(con, -1, 0)

    bounds = problem["Bounds"]
    #x = np.array([1,2])
    #A = np.array([1,0])
    #B = np.array([0,1])
    #print(A.dot(x),B.dot(x))
    #print(bounds)
    #LinearConstraint(A,bounds[0], bounds[1], keep_feasible=True)} # Not for cobula??
    
    ## COBYLA only supports inequality constraints
    
    constraint_list = [{"type": "ineq","fun": lambda x, idx = i:( (1-2*(idx%2))*x[(idx//2)] )+ bounds[idx]} for i in range(len(bounds))] ## found some stupid shit about lambdas in list comprehenssion here... covered nicely in https://stackoverflow.com/a/34021333/21573210
    constraint_list.extend([{"type": "ineq","fun": lambda x: 0.5 - ( constraintf[j](x[None]) )[0] } for j in range(len(constraintf))])

    # constraints = [{"type": "ineq","fun": lambda x: x[0] + bounds[0]},
    #                      {"type": "ineq","fun": lambda x: -x[0] + bounds[1]},
    #                      {"type": "ineq","fun": lambda x: x[1] + bounds[0]},
    #                      {"type": "ineq","fun": lambda x: -x[1] + bounds[1]},
    #                      {"type": "ineq","fun": lambda x: 0.5-con(x)[0]}]
    
    # xxx = np.array([6.1,-1])
    # for i in range(len(constraint_list)):
    #     #print([1,2,3,4][(i//2)])
    #     print("\n", constraint_list[i]["fun"](xxx), constraints[i]["fun"](xxx))
    #     print(bounds[i])
    
    def callbackF(Xi, list_in): # For storing values of x traversed in the algorithm
        list_in.append(Xi)
        return False
        
    trace = []
    callBF = lambda x: callbackF(x,trace)
    opt = scipy.optimize.minimize(fun, x0, callback = callBF, constraints = constraint_list, 
                                  method="COBYLA", tol = obj_tol, options = {"maxiter":maxiter}) #,tol = 0.1 #catol does not work???
    xs_out = np.array(trace)
    
    #Also a lot of infeasable solutions in eikson paper
    x = opt.x
    print(opt.message)
    print(x,fun(x))
    
    constr_out = np.array([constraintf[i](xs_out) for i in range(len(constraintf))]).T
    objs_out = np.concatenate((costf(xs_out).reshape(-1,1), constr_out),axis = 1)
    all_objs = True
    print(xs_out)
    print(objs_out)

    return xs_out, objs_out, all_objs


if __name__ == "__main__":
    from opt_problems.ADMMBO_paper_problems import gardner1
    
    for i in range(10):
        bounds = gardner1["Bounds"]
        rnd = np.random.RandomState(42 + i)
        x0 = ((rnd.rand(2)*(bounds[1]-bounds[0])))+bounds[0]
        print("-----\n",x0)
        cobyla_run(gardner1, x0)