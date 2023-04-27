import numpy as np

from ADMMBO_scaffold import admmbo_run
from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run, multi_cobyla
from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from utils.plotting import vizualize_toy
from opt_problems.paper_problems import gardner1, gardner2, gramacy
from utils.sampling import monte_carlo_sampling, grid_sampling
from utils.storing import create_exp_folder, save_exps

# For testting
xs, objs, indiv_evals = None, None, None


#xs, objs, indiv_evals = cobyla_run(problem, x0)


problem = gardner1
name = "gard1-testing"
pesc_create_problem(problem, name, decoupled=True)
pesc_run_experiment(name, max_iter=100)
#
x0s = monte_carlo_sampling(problem, num = 100, seed = 12) #np.array([[1.5,4.5],[1,2]])
x0 = x0s[0]

alg_res = [
    #multi_cobyla(problem, x0s),
    #pesc_main(name = name),
    #cma_es(problem, x0),
    admmbo_run(problem, x0s, start_all = False)
]
if xs != None:
    vizualize_toy(xs, objs, indiv_evals, problem)

for n, res in enumerate(alg_res):
    xs, objs, indiv_evals = res
    vizualize_toy(xs,objs, indiv_evals ,problem)
    #save_exps([res], names[n], e_folder=e_folder)
