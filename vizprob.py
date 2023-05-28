from opt_problems.paper_problems import gardner1,gardner2,gramacy, lamwillcox3
from opt_problems.example_problems import example0

from utils.plotting import vizualize_toy_problem

vizualize_toy_problem(gardner1, points_per_dim=1000, name= "Gardner 1")
vizualize_toy_problem(gardner2, points_per_dim=1000, name= "Gardner 2")
vizualize_toy_problem(gramacy, points_per_dim=1000, name= "Gramacy")
vizualize_toy_problem(example0, points_per_dim=1000, name= "Flower")
vizualize_toy_problem(lamwillcox3, points_per_dim=40)