import numpy as np
from time import time
import warnings

from ADMMBO_scaffold import admmbo_run
from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import cobyla_run, multi_cobyla

from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from utils.plotting import vizualize_toy
from utils.sampling import monte_carlo_sampling, grid_sampling, noisy_grid, latin_grid
from utils.storing import create_exp_folder, save_exps


from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3, gramsingle, lwwrong
from opt_problems.example_problems import example0
try:  
    from opt_problems.coil import coil
    from simhack.prob_coil import coil_pure
except:
    pass

warnings.filterwarnings('ignore')

running_time = 6*60*60

exp_name = "grid-coil-360" #"PESC-LW"
num_trials = 10 #60 
problem = coil_pure #coil_pure
name = None#"example0" #"lw3" #For PESC

max_iter = 360#120

alg_res = { 
            #"cobyla":[],
            #"cma":[],
            #"pesc":[],
            "admmbo": []
}


divisor_iter = 1 + len(problem["Constraint Functions (z)"]) #Half for cma and cobyla since they are coupled
if not name is None:
    pesc_create_problem(problem, name, decoupled=True, max_iter = max_iter)
else:
    _ = alg_res.pop("pesc", None)

#ops00 = {"M": 20, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":8}
# ops0 = {"M": 10, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4}

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

rho_sub = {
    "Rho-1" :            {"M": 10, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "(Locked)_Rho-0.1" : {"M": 10, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
}

M_testing = {
    "M-0.1_Rho-1" :            {"M": 0.1, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-0.1_(Locked)_Rho-0.1" : {"M": 0.1, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-1_Rho-1" :            {"M": 1, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-1_(Locked)_Rho-0.1" : {"M": 1, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-10_Rho-1" :            {"M": 10, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-10_(Locked)_Rho-0.1" : {"M": 10, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-20_Rho-1" :            {"M": 30, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-20_(Locked)_Rho-0.1" : {"M": 30, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-100_Rho-1" :            {"M": 100, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-100_(Locked)_Rho-0.1" : {"M": 100, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-1000_Rho-1" :            {"M": 1000, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
    "M-1000_(Locked)_Rho-0.1" : {"M": 1000, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2, "beta0":4},
}

B_mult = 2 #Double the constraints test could be good??

coil_testing = {
    "Rho-1" : {"M": 20, "adjust_rho" : True, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "(Locked)_Rho-0.1_M-20" : {"M": 20, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "Rho-0.1" : {"M": 20, "adjust_rho" : True, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "(Locked)_Rho-1" : {"M": 20, "adjust_rho" : False, "rho" : 0.5, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult}
}

B_mult = 1
coil_testing2 = {
    #"_M-1e5_Rho-1" : {"M": 1e5, "adjust_rho" : True, "rho" :1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "_M-1e5(Locked)_Rho-1" : {"M": 1e5, "adjust_rho" : False, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "_M-1e5(Locked)_Rho-2" : {"M": 1e5, "adjust_rho" : False, "rho" : 2, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "_M-1e7(Locked)_Rho-1" : {"M": 1e7, "adjust_rho" : False, "rho" : 1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    "_M-1e8(Locked)_Rho-0.1" : {"M": 1e8, "adjust_rho" : False, "rho" : 0.1, "epsilon" : 0, "alpha": 2, "alpha0": 4, "beta": 2*B_mult, "beta0":4*B_mult},
    #"Budget_M-1e5(Locked)_Rho-1" : {"M": 1e5, "adjust_rho" : False, "rho" : 1, "epsilon" : 0, "alpha": 4, "alpha0": 8, "beta": 4*B_mult, "beta0":8*B_mult},
}

admmbo_opts = coil_testing2
grid = True
prop_grid = 0.6/divisor_iter #Doubly sampled points at start...
pre = "LATIN_" #"GRID_"# "Grid_"#""

if "admmbo" in alg_res.keys():
    for name_addon in admmbo_opts.keys():
        alg_res[pre +"admmbo" + name_addon] = []

s_time = time()
for e_num in range(num_trials):
    x0s = monte_carlo_sampling(problem, num = max_iter, seed = 14 + e_num)
    if grid:
        bounds = problem["Bounds"]
        dim_num = len(bounds)//2
        if pre[0].lower() == "g":
            x0s = noisy_grid(problem, num_per_dim = int((prop_grid*max_iter)**(1/dim_num)), seed = 14 + e_num)
        if pre[0].lower() == "l":
            x0s = latin_grid(problem, num = int(prop_grid*max_iter), seed = 14 + e_num)
        else:
            raise NotImplementedError("Not correct pre formatting")
        
    x0 = x0s[0]

    if "admmbo" in alg_res.keys():
        for name_addon, opt_dict in admmbo_opts.items():
            alg_res[pre + "admmbo" + name_addon].append(admmbo_run(problem, x0s, start_all = grid,                                                              
                                                             max_iter=max_iter, admmbo_pars=opt_dict))    
    if "pesc" in alg_res.keys():
        pesc_run_experiment(name, max_iter=120)
        alg_res["pesc"].append(pesc_main(name))
    if "cma" in alg_res.keys():
        alg_res["cma"].append(cma_es(problem, x0, max_iter = 0.9*max_iter//divisor_iter)) #I give it some extra to get feasable (should be done the other way around)
    if "cobyla" in alg_res.keys():
        alg_res["cobyla"].append(multi_cobyla(problem, x0s, maxiter_total=max_iter//divisor_iter))

    
    mins, _ = divmod(time() - s_time, 60)
    hours, mins = divmod(mins, 60)
    print("\n", "-"*40,"\n", f"Finished running iteration {e_num + 1} after {int(hours):02d}h {int(mins):02d}m")
    if time() - s_time > running_time:
        print("\n", "-"*40,f"\nTerminating at it {e_num + 1} after set running time\n", "-"*40,"\n",)
        break

if len(admmbo_opts) > 0:
    _ = alg_res.pop("admmbo", None)

e_folder = create_exp_folder(exp_name)
for alg_name, res_list in alg_res.items():
    save_exps(res_list, alg_name, e_folder=e_folder)

