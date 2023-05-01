import scipy
import scipy.optimize as opt
import numpy as np
import warnings

class OutOfBounds(Warning): #A thanks to answer https://stackoverflow.com/a/70739166/21573210 for the way to stop the iterating!
    pass

def cobyla_run(problem, x0, obj_tol = 0.1, maxiter = 100, multi = False):
    assert x0.ndim == 1, "can't start with multiple 0 points??"
    
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    fun = lambda x: costf(x[None]) # Function in to minimize

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

    def callbackF(Xi, list_in): # For storing values of x traversed in the algorithm and stopping if out of bounds
        list_in.append(Xi)
        for i in range(len(bounds)):
            if constraint_list[i]["fun"](Xi) < 0:
                warnings.warn("Terminating optimization: Sampled out of bounds point!!",
                          OutOfBounds)
                return True #Failed, and sampled outside of bounds #This should have ended the iterations?? Above solution does
        return False
        
    trace = [x0] #TODO: Check that this also samples x0 at the start, as well as cma-es
    callBF = lambda x: callbackF(x,trace)
    res = scipy.optimize.minimize(fun, x0, callback = callBF, constraints = constraint_list, 
                                  method="SLSQP", tol = obj_tol, options = {"maxiter":maxiter, "catol":0}) #,tol = 0.1 #catol does not work???
    xs_out = np.array(trace)
    
    constr_out = np.array([constraintf[i](xs_out) for i in range(len(constraintf))]).T
    objs_out = np.concatenate((costf(xs_out).reshape(-1,1), constr_out),axis = 1)
    indiv_eval = []

    if multi:
        return xs_out, objs_out, indiv_eval

    #Also a lot of infeasable solutions in eikson paper
    x = res.x
    # print(res.message)
    # print(x,fun(x))
    # print(res.keys())
    return xs_out, objs_out, indiv_eval

def multi_cobyla(problem, x0s, obj_tol= 0.1, maxiter_per = None, maxiter_total = 100, divide = True):
    it = 0
    budget_left = maxiter_total
    xs_list, objs_list = [], []
    print("COBYLA: ", end = "", flush=True)
    while budget_left > 0 and it < len(x0s):        
        prct_done = 100*(1-(budget_left/maxiter_total))
        print(f"{prct_done:.2f}%|", end="", flush = True)
        xs_out, objs_out, indiv_eval = cobyla_run(problem,x0s[it], maxiter=budget_left, multi=True)
        it += 1
        budget_left -= len(xs_out)
        xs_list.append(xs_out)
        objs_list.append(objs_out)
        
    tot_xs = np.concatenate(xs_list, axis = 0)
    tot_objs = np.concatenate(objs_list, axis = 0)
    
    return tot_xs, tot_objs, []


if __name__ == "__main__":
    from opt_problems.paper_problems import gardner1
    
    for i in range(10):
        bounds = gardner1["Bounds"]
        rnd = np.random.RandomState(42 + i)
        x0 = ((rnd.rand(2)*(bounds[1]-bounds[0])))+bounds[0]
        print("-----\n",x0)
        cobyla_run(gardner1, x0)