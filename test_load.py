import matplotlib.pyplot as plt
import os

from utils.storing import fol, load_exp
from utils.plotting import * #expretiment_plot, vizualize_toy

from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3
from opt_problems.example_problems import example0
#from opt_problems.coil import coil
#from simhack.prob_coil import coil_pure

exclude = ["plot_cache"] #["admmbo00","admmbo01"]
e_folder = fol("rho-gram", 0)
problem = gramacy#coil_pure


"""
e_folder = fol("lw3-all", 0)
problem = lamwillcox3   
"""

print(e_folder)

name_from_to = {'admmbo':'ADMMBO','_': ' ', '-': " = ", ' =  = ': " = -",
                'cma': 'CMA-ES', 'cobyla': 'COBYLA (Multiple)', 'pesc':'PESC',
                }

alg_folders = os.listdir(e_folder)
for excluded in exclude:
    if not excluded in alg_folders:
        continue
    alg_folders.remove(excluded)
exps = {}

## Selecting algs by indx (optional) ##
#alg_folders = alg_folders[6:10] + alg_folders[-3:]
## -------------------------------   ##
for folder in alg_folders:
    exps[folder] = load_exp(folder,e_folder)

expretiment_plot(exps,problem,e_folder,name_from_to=name_from_to, override=True)

for name, exp_list in exps.items():
    continue
    exploration_hist(exp_list, name = name)

plt.show()
# a,b,c = exps["ADMMBO"][61]
# vizualize_toy(a,b,c, problem)
# expretiment_plot({"test" : exps["ADMMBO"][60:62]},problem)