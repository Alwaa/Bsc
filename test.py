import numpy as np

from ADMMBO_scaffold import admmbo_run
from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run
from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from plotting import vizualize_toy
from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
from utils.sampling import monte_carlo_sampling, grid_sampling


# For testting
xs = None

#If all_objective_values is not true, then it returns eval type array (maybe not the best formatting)

#xs, objs, all_objective_values = cma_es(problem)


#xs, objs, all_objective_values = cobyla_run(problem, x0)
# cobyla_run(problem, x0)



# pesc_create_problem(gardner1, "test3")
# pesc_run_experiment("test3")
# xs, objs, all_objective_values = pesc_main(name = "test3")

problem = gardner2
name = "test3"

x0s = np.array([[1.5,4.5],[1,2]])
x0 = x0s[0]
alg_res = [
    cobyla_run(problem, x0),
    #pesc_main(name = name),
    cma_es(problem, x0),
    admmbo_run(problem, x0s)
]
if xs != None:
    vizualize_toy(xs, objs, indiv_evlas, problem)

print(len(alg_res),alg_res[0])

for res in alg_res:
    xs, objs, indiv_evals = res
    vizualize_toy(xs,objs, indiv_evals ,problem)


print(monte_carlo_sampling(gardner1, num = 4))
print(grid_sampling(gardner1))