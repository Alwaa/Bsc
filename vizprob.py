from opt_problems.paper_problems import gardner1,gardner2,gramacy
from opt_problems.example_problems import example0

from utils.plotting import vizualize_toy_problem

vizualize_toy_problem(gardner1, points_per_dim=400, name= "Gardner 1")
vizualize_toy_problem(gardner2, points_per_dim=400, name= "Gardner 2")
vizualize_toy_problem(gramacy, points_per_dim=400, name= "Gramacy")
vizualize_toy_problem(example0, points_per_dim=400)