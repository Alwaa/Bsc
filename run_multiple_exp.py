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
#from opt_problems.coil import coil
#from simhack.prob_coil import coil_pure

warnings.filterwarnings('ignore')

running_time = 13*60*60

exp_name = "rho-gard2"
num_trials = 60
problem = gardner2 #gardner1 #lamwillcox3 #gramacy
name = "gard2" #"coil-test" #For PESC

max_iter = 120 #PESC and ADMMBO 
divisor_iter = 2 #Half for cma and cobyla since they are coupled
#pesc_create_problem(problem, name, decoupled=True, max_iter = max_iter)

alg_res = { 
            "cobyla":[],
            "cma":[],
            "pesc":[],
            "admmbo": []
}

#ops00 = {"M": 20, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":8}
# ops0 = {"M": 10, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4}
#TODO: Run big M coil test again since i didnt configure correctly...

rho_testing = {
    "Rho-0.05" : {"M": 10, "rho" : 0.05, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "Rho-0.1" : {"M": 10, "adjust_rho" : True, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "Rho-0.2" : {"M": 10, "adjust_rho" : True, "rho" : 0.2, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "Rho-0.5" : {"M": 10, "adjust_rho" : True, "rho" : 0.5, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "Rho-1" : {"M": 10, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-0.05" : {"M": 10, "adjust_rho" : False, "rho" : 0.05, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-0.1" : {"M": 10, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-0.2" : {"M": 10, "adjust_rho" : False, "rho" : 0.2, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-0.5" : {"M": 10, "adjust_rho" : False, "rho" : 0.5, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-1" : {"M": 10, "adjust_rho" : False, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
}

admmbo_opts = rho_testing #{}


if "admmbo" in alg_res.keys():
    for name_addon in admmbo_opts.keys():
        alg_res["admmbo" + name_addon] = []

s_time = time()
for e_num in range(num_trials):
    x0s = monte_carlo_sampling(problem, num = max_iter, seed = 14 + e_num)
    x0 = x0s[0]

    if "admmbo" in alg_res.keys():
        for name_addon, opt_dict in admmbo_opts.items():
            alg_res["admmbo" + name_addon].append(admmbo_run(problem, x0s, start_all = False,                                                              max_iter=max_iter, admmbo_pars=opt_dict))    
    if "pesc" in alg_res.keys():
        pesc_run_experiment(name, max_iter=120)
        alg_res["pesc"].append(pesc_main(name))
    if "cma" in alg_res.keys():
        alg_res["cma"].append(cma_es(problem, x0, max_iter = max_iter//divisor_iter))
    if "cobyla" in alg_res.keys():
        alg_res["cobyla"].append(multi_cobyla(problem, x0s, maxiter_total=max_iter//divisor_iter))

    
    mins, _ = divmod(time() - s_time, 60)
    hours, mins = divmod(mins, 60)
    print("\n", "-"*40,"\n", f"Finished running iteration {e_num + 1} after {int(hours):02d}h {int(mins):02d}m")
    if time() - s_time > running_time:
        print("\n", "-"*40,"\nTerminating runs after set running time\n", "-"*40,"\n",)
        break

if len(admmbo_opts) > 0:
    _ = alg_res.pop("admmbo")

e_folder = create_exp_folder(exp_name)
for alg_name, res_list in alg_res.items():
    save_exps(res_list, alg_name, e_folder=e_folder)

