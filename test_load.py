import matplotlib.pyplot as plt

from utils.storing import fol, load_exp
from utils.plotting import expretiment_plot

from opt_problems.paper_problems import gardner1, gardner2, gramacy
from opt_problems.example_problems import example0


e_folder = fol("lw-dec-test", 1)
problem = example0

print(e_folder)

exps = {
    "ADMMBO": load_exp("admmbo",e_folder),
#    "PESC": load_exp("pesc",e_folder),
    "CMA-ES": load_exp("cma",e_folder),
    "COBYLA (Multiple)": load_exp("cobyla",e_folder)
}
print(len(exps["ADMMBO"]))
expretiment_plot(exps,problem)

plt.show()