import matplotlib.pyplot as plt

from utils.storing import fol, load_exp
from utils.plotting import expretiment_plot, vizualize_toy

from opt_problems.paper_problems import gardner1, gardner2, gramacy, lamwillcox3
from opt_problems.example_problems import example0


e_folder = fol("gram-dec-all", 2)
problem = gramacy

print(e_folder)

exps = {
    "ADMMBO": load_exp("admmbo",e_folder),
    "PESC": load_exp("pesc",e_folder),
    "CMA-ES": load_exp("cma",e_folder),
    "COBYLA (Multiple)": load_exp("cobyla",e_folder)
}
print(len(exps["ADMMBO"]))
expretiment_plot(exps,problem)

plt.show()
# a,b,c = exps["ADMMBO"][61]
# vizualize_toy(a,b,c, problem)
# expretiment_plot({"test" : exps["ADMMBO"][60:62]},problem)