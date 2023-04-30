import cma
import matplotlib.pyplot as plt
from numpy import r_
import numpy as np

# notebooks and the (messy) documentation used for reference


#es.optimize(cma.ff.rosen)
#es.result_pretty()

def cma_es(problem, x0, max_iter = 120):
    own_logger_X = []
    own_logger_fit = []
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    ## Differently formated input than my current ADDMBO impl
    ## actually closer to original that can take both?
    def constraints(x):
        return [0.5 - constraintf[i](x[None])[0] for i in range(len(constraintf))] #I think this is the correct format??
    def fun(x):
        return costf(x[None])[0]

    bounds = [problem["Bounds"][0::2],problem["Bounds"][1::2]] #I assume this works as per the documentation
    opts = {"bounds" : bounds, 
            "maxfevals" : max_iter}

    cfun = cma.ConstrainedFitnessAL(fun, constraints)
    sigma0 = 1    # initial standard deviation to sample new solutions

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts) #TODO: Double check that it also samples the first point
    own_logger_X.append(x0)
    while not es.stop():
        X, fit = es.ask_and_eval(cfun)  # sample len(X) candidate solutions
        es.tell(X, fit)
        own_logger_X.extend(X)
        own_logger_fit.extend(fit) # I can get in the middle and log if i want
        cfun.update(es)
        es.logger.add()  # for later plotting
        es.disp()
    x = es.result.xfavorite  # the original x-value may be meaningless
    #print("\n-------\n", x)
    #print(constraints(x))  # show constraint violation values

    #### Solution could be not feasable #####
    c = es.countiter
    new_max_evals = int(opts["maxfevals"]*1.2)
    es.opts.set({"maxfevals" : new_max_evals})
    x = cfun.find_feasible(es)
    if x is None:
        pass
    else:
        #print("find_feasible took {} iterations".format(es.countiter - c))
        #TODO:Add padding of points equivalent to new points sampled and then add the feasable point
        #print("\n-------\n", x)
        #constraints(x)  # is now <= 0
        #### ------------------------------ #####

        es.result_pretty()

        # cma.plot()
        # cma.s.figshow()
        # plt.show(block=True) ## Stops VS Code from closing it

        #cma.disp(None, np.r_[0:int(1e9):10, -1]) # every 10-th and last
        #cma.disp(name = 'outcma/xrecentbest', idx = np.r_[0:int(1e9):10, -1])

        # print("\n\n",x)
        # print(fun(x),"\n\n")
        # print(len(cfun.archives[1].archive))
        
        # assume some data are available from previous runs
        cma.disp(None, r_[0:int(1e9),-1])
        #print(cma.CMAOptions("verb"))
        if False: #print options in prettier format (can input string for suboptions)
            for k,v in cma.CMAOptions().items():
                print(k, v)
        
        own_logger_X.append(x)
    x_out = np.array(own_logger_X)
    
    constr_out = np.array([constraintf[i](x_out) for i in range(len(constraintf))]).T
    objs_out = np.concatenate((costf(x_out).reshape(-1,1), constr_out),axis = 1)
    indiv_eval = []
    return x_out, objs_out, indiv_eval
    #### TODO: Clean up a bit
    
if __name__ == "__main__":
    from opt_problems.paper_problems import gardner1
    from utils.plotting import vizualize_toy
    print(gardner1)
    x0 = 2 * [2]  # initial sample 
    a,b,c = cma_es(gardner1, x0)
    vizualize_toy(a,b, c,gardner1)