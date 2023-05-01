import matplotlib.pyplot as plt
import os

from utils.storing import fol, load_exp
from utils.plotting import expretiment_plot, vizualize_toy

from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3
from opt_problems.example_problems import example0
from opt_problems.coil import coil
from simhack.prob_coil import coil_pure

exclude = ["plot_cache"] #["admmbo00","admmbo01"]
e_folder = fol("coil-test", 1)
problem = coil_pure


"""
e_folder = fol("lw3-all", 0)
problem = lamwillcox3   
"""

print(e_folder)

name_from_to = {'admmbo':'ADMMBO', 'cma': 'CMA-ES', 'cobyla': 'COBYLA (Multiple)', 'pesc':'PESC',
                '_M': ' M = '}

alg_folders = os.listdir(e_folder)
for excluded in exclude:
    if not excluded in alg_folders:
        continue
    alg_folders.remove(excluded)

exps = {}
for folder in alg_folders:
    exps[folder] = load_exp(folder,e_folder)

print(len(exps[alg_folders[0]]))
expretiment_plot(exps,problem,e_folder,name_from_to=name_from_to)

plt.show()
# a,b,c = exps["ADMMBO"][61]
# vizualize_toy(a,b,c, problem)
# expretiment_plot({"test" : exps["ADMMBO"][60:62]},problem)