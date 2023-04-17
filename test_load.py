from utils.storing import fol, load_exp
from utils.plotting import expretiment_plot

from opt_problems.paper_problems import gardner1, gramacy


e_folder = fol("test", 4)
problem = gramacy
alg_name = "cobyla"

print(e_folder)
exp_list = load_exp(alg_name,e_folder)

expretiment_plot(exp_list,problem)