import matplotlib.pyplot as plt

from utils.storing import fol, load_exp
from utils.plotting import expretiment_plot

from opt_problems.paper_problems import gardner1, gramacy


e_folder = fol("testing", 1)
problem = gramacy
alg_name = "admmbo"

print(e_folder)

exps = {
    "ADMMBO": load_exp("admmbo",e_folder),
    "CMA-ES": load_exp("cma",e_folder),
    "COBYLA (Multiple)": load_exp("cobyla",e_folder)
}

expretiment_plot(exps,problem)

plt.show()