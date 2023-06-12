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

"""
Script for loading and plotting resutls,
 not a generally recomended format as there are several places to adjust the code when switching between experiments 
"""

exclude = ["plot_cache"] #["admmbo00","admmbo01"] , "cma","cobyla"
e_folder = fol("grid-coil-360",1)#fol("grid-coil-MM", 0)###fol("tsts",3)#fol("rho-lw-wrong",0) #fol("coil-more",0)#fol("coil-test", 2)
problem = coil_pure #lwwrong# coil_pure
name_title = "Coil"
title = "Comparison " + name_title #"Comparison Gardner 1"


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
error_bars = True
#alg_folders = alg_folders[1::8] + alg_folders[-3:]
#alg_folders = alg_folders[1:2] + alg_folders[-3:]
#alg_folders = alg_folders[:-3]
#alg_folders = list(np.array(alg_folders)[np.array([0,2])])
show_indxs = []
## -------------------------------   ##
for folder in alg_folders:
    exps[folder] = load_exp(folder,e_folder)

expretiment_plot(exps,problem,e_folder,name_from_to=name_from_to, 
                 override=False, title=title, just_mean=not error_bars,
                 xlim= 360, split_starting = True)

for name, exp_list in exps.items():
    continue
    exploration_hist(exp_list, name = name)

plt.show()
for i in show_indxs:
    a,b,c = exps[alg_folders[i]][2]
    name = alg_folders[i]
    for nfrom, nto in name_from_to.items():
        name = name.replace(nfrom,nto)
    vizualize_toy(a,b,c, problem
    ,title = f"{name_title} Path :{name}")

    # a,b,c = exps[alg_folders[3]][2]
    # vizualize_toy(a,b,c, problem
    # ,title = "Flower PESC Path")
# expretiment_plot({"test" : exps["ADMMBO"][60:62]},problem)