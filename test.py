import numpy as np

from ADMMBO_scaffold import admmbo_run
from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run, multi_cobyla
from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from utils.plotting import vizualize_toy
from opt_problems.paper_problems import gardner1, gardner2, gramacy
from opt_problems.example_problems import example0
from utils.sampling import monte_carlo_sampling, grid_sampling, noisy_grid, latin_grid
from utils.storing import create_exp_folder, save_exps

# For testting
xs, objs, indiv_evals = None, None, None


#xs, objs, indiv_evals = cobyla_run(problem, x0)


problem = example0 #gardner1
name = "gard1-testing"
#pesc_create_problem(problem, name, decoupled=True)
#pesc_run_experiment(name, max_iter=100)
#
x0s = monte_carlo_sampling(problem, num = 100, seed = 12) #np.array([[1.5,4.5],[1,2]])
x0 = x0s[0]

x0s = grid_sampling(problem, num_per_dim=5)
bounds = problem["Bounds"]
dim_num = len(bounds)//2


x0s = noisy_grid(problem, num_per_dim=int(((100/3)**(1/dim_num))), seed = 12)

x0s = latin_grid(problem, num= 40, seed = 12)
print(x0s)

alg_res = [
    #multi_cobyla(problem, x0s),
    #pesc_main(name = name),
    #cma_es(problem, x0),
    admmbo_run(problem, x0s, start_all = True)
]
if xs != None:
    vizualize_toy(xs, objs, indiv_evals, problem)

for n, res in enumerate(alg_res):
    xs, objs, indiv_evals = res
    vizualize_toy(xs,objs, indiv_evals ,problem)
    #save_exps([res], names[n], e_folder=e_folder)
