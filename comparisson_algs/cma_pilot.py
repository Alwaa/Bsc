import cma
import matplotlib.pyplot as plt
from numpy import r_

# notebooks and the (messy) documentation used for reference


#es.optimize(cma.ff.rosen)
#es.result_pretty()

def cma_es(problem):
    own_logger = []
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    ## Differently formated input than my current ADDMBO impl
    ## actually closer to original that can take both?
    def constraints(x):
        #return constraintf[0](x[None])
        return [constraintf[i](x[None])[0] -0.5 for i in range(len(constraintf))] #I think this is the correct format??
    def fun(x):
        #print("TEST\n",x,"\n\n")
        #print(costf(x[None]))
        return costf(x[None])[0]

    #tolx = 0.01
    #tolfun
    #maxfevals
    #bounds = [for_all,for_all] or [[1,2,3],for all]
    if problem["Bound Type"] == "square":
        bounds = [problem["Bounds"][0],problem["Bounds"][1]]
    print(bounds)
    opts = {"bounds" : bounds, 
            "maxfevals" : 100}

    cfun = cma.ConstrainedFitnessAL(fun, constraints)
    x0 = 2 * [2]  # initial solution
    sigma0 = 1    # initial standard deviation to sample new solutions

    #x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X, fit = es.ask_and_eval(cfun)  # sample len(X) candidate solutions
        es.tell(X, fit)
        own_logger.append((X,fit)) # I can get in the middle and log if i want
        cfun.update(es)
        es.logger.add()  # for later plotting
        es.disp()
    x = es.result.xfavorite  # the original x-value may be meaningless
    print("\n-------\n", x)
    print(constraints(x))  # show constraint violation values
    constraints(x)

    #### Solution could be not feasable #####
    c = es.countiter
    x = cfun.find_feasible(es)
    print("find_feasible took {} iterations".format(es.countiter - c))
    print("\n-------\n", x)
    constraints(x)  # is now <= 0
    #### ------------------------------ #####

    es.result_pretty()

    # cma.plot()
    # cma.s.figshow()
    # plt.show(block=True) ## Stops VS Code from closing it

    #cma.disp(None, np.r_[0:int(1e9):10, -1]) # every 10-th and last
    #cma.disp(name = 'outcma/xrecentbest', idx = np.r_[0:int(1e9):10, -1])

    print("\n\n",x)
    print(fun(x),"\n\n")
    
    print(len(cfun.archives[1].archive))
    
    # assume some data are available from previous runs
    cma.disp(None, r_[0:int(1e9),-1])
    #print(cma.CMAOptions("verb"))
    if False: #print options in prettier format (can input string for suboptions)
        for k,v in cma.CMAOptions().items():
            print(k, v)
    print(own_logger)
    return cfun
    #### TODO: Clean up a bit, then maybe try the other CMA-ES python IMPL since it has nicer plotting
    
if __name__ == "__main__":
    from opt_problems.ADMMBO_paper_problems import gardner1
    print(gardner1)
    cma_es(gardner1)