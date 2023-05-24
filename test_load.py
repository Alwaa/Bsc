import matplotlib.pyplot as plt
import os

from utils.storing import fol, load_exp
from utils.plotting import * #expretiment_plot, vizualize_toy

from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3, lwwrong, gramsingle
from opt_problems.example_problems import example0
try:  
    from opt_problems.coil import coil
    from simhack.prob_coil import coil_pure
except:
    pass

exclude = ["plot_cache"] #["admmbo00","admmbo01"]
e_folder = fol("yz-gard1",2)#fol("rho-lw-wrong",0) #fol("coil-more",0)#fol("coil-test", 2)
problem = gardner1#gramsingle#lwwrong#lamwillcox3#example0# coil_pure
title = "Comparison Gardner 1"#"Comparison Gramacy Single"
error_bars = True

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
#alg_folders = alg_folders[1::8] + alg_folders[-3:]
#alg_folders = list(np.array(alg_folders)[np.array([1,3,7])])
## -------------------------------   ##
for folder in alg_folders:
    exps[folder] = load_exp(folder,e_folder)

expretiment_plot(exps,problem,e_folder,name_from_to=name_from_to, 
                 override=True, title=title, just_mean=not error_bars)

for name, exp_list in exps.items():
    continue
    exploration_hist(exp_list, name = name)

plt.show()
a,b,c = exps[alg_folders[0]][2]
vizualize_toy(a,b,c, problem)

a,b,c = exps[alg_folders[1]][2]
vizualize_toy(a,b,c, problem)
# expretiment_plot({"test" : exps["ADMMBO"][60:62]},problem)