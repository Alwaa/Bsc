import numpy as np

from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run
from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from plotting import vizualize_toy
from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
problem = gardner1

# For testting
xs = None


#xs, objs, all_objective_values = cma_es(problem)


# NOT CORRECT OUTPUT YET !!! #xs, objs, all_objective_values = COBYLA(problem)
#cobyla_run(problem, x0)



#pesc_create_problem(gardner1, "test3")
# pesc_run_experiment("test3")
# xs, objs, all_objective_values = pesc_main(name = "test3")


if xs != None:
    vizualize_toy(xs,objs,problem,decoupled=False)



def monte_carlo_sampling(problem, num = 1): #Return list of random ints, not applicable in most algs, as onlt one start point is allowed...
    bounds = problem["Bounds"]
    dim_num = len(bounds)//2
    
    x0s = np.zeros((num, dim_num))
    
    rnd = np.random.RandomState(42 + 1)
    for i in range(dim_num):
        x0s[:,i] = rnd.uniform(bounds[2*i], bounds[2*i + 1], num)
    return x0s

print(monte_carlo_sampling(gardner1, num = 4))