import numpy as np
from time import time
import warnings

from ADMMBO_scaffold import admmbo_run
from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run, multi_cobyla

from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from utils.plotting import vizualize_toy
from utils.sampling import monte_carlo_sampling, grid_sampling
from utils.storing import create_exp_folder, save_exps


from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3
from opt_problems.example_problems import example0

warnings.filterwarnings('ignore')

running_time = 14*60*60

exp_name = "gram-dec-all" +"redo"
num_trials = 64
problem = gramacy #lamwillcox3 #gramacy
name = "gramacy-dectest1" #For PESC

max_iter = 120 #PESC and ADMMBO double for cma and cobyla
#pesc_create_problem(gramacy, name, decoupled=True, max_iter = max_iter)

alg_res = { 
            # "cobyla":[],
            # "cma":[],
            # "pesc":[],
            "admmbo": []
}

s_time = time()
for e_num in range(num_trials):
    x0s = monte_carlo_sampling(problem, num = max_iter, seed = 14 + e_num)
    x0 = x0s[0]
    
    if "pesc" in alg_res.keys():
        pesc_run_experiment(name)
        alg_res["pesc"].append(pesc_main(name))
    if "admmbo" in alg_res.keys():
        alg_res["admmbo"].append(admmbo_run(problem, x0s, start_all = False, max_iter=max_iter))
    if "cma" in alg_res.keys():
        alg_res["cma"].append(cma_es(problem, x0, max_iter = max_iter*2))
    if "cobyla" in alg_res.keys():
        alg_res["cobyla"].append(multi_cobyla(problem, x0s, maxiter_total=max_iter*2))
    
    mins, _ = divmod(time() - s_time, 60)
    hours, mins = divmod(mins, 60)
    print("\n", "-"*40,"\n", f"Finished running iteration {e_num + 1} after {int(hours):02d}h {int(mins):02d}m")
    if time() - s_time > running_time:
        print("\n", "-"*40,"\nTerminating runs after set running time\n", "-"*40,"\n",)
        break
    
e_folder = create_exp_folder(exp_name)
for alg_name, res_list in alg_res.items():
    save_exps(res_list, alg_name, e_folder=e_folder)

