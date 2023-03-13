import cma
import matplotlib.pyplot as plt

# notebooks and the (messy) documentation used for reference


#es.optimize(cma.ff.rosen)
#es.result_pretty()

def cma_es(problem):
    es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Function (z)"]
    ## Differently formated input than my current ADDMBO impl
    ## actually closer to original that can take both?
    def constraints(x):
        #print(constraintf[0](x[None]))
        return constraintf[0](x[None])
    def fun(x):
        #print("TEST\n",x,"\n\n")
        #print(costf(x[None]))
        return costf(x[None])


    cfun = cma.ConstrainedFitnessAL(fun, constraints)
    x0 = 2 * [2]  # initial solution
    sigma0 = 1    # initial standard deviation to sample new solutions

    x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)
    x = es.result.xfavorite  # the original x-value may be meaningless
    constraints(x)  # show constraint violation values

    #### Solution could be not feasable #####
    c = es.countiter
    x = cfun.find_feasible(es)
    print("find_feasible took {} iterations".format(es.countiter - c))
    constraints(x)  # is now <= 0
    #### ------------------------------ #####

    es.result_pretty()

    cma.plot()
    cma.s.figshow()
    plt.show(block=True) ## Stops VS Code from closing it

    #cma.disp(None, np.r_[0:int(1e9):10, -1]) # every 10-th and last
    #cma.disp(name = 'outcma/xrecentbest', idx = np.r_[0:int(1e9):10, -1])

    print("\n\n",x)
    print(fun(x),"\n\n")

    #### TODO: Clean up a bit, then maybe try the other CMA-ES python IMPL since it has nicer plotting