import numpy as np

# http://proceedings.mlr.press/v32/gardner14.pdf
# Originally from Jacob R Gardner, Matt J Kusner, Zhixiang Eddie Xu, Kilian Q Weinberger, and John P Cunningham. Bayesian optimization with inequality constraints.

#--------------------------------#
# Bounds and such
#--------------------------------#

boundtype_gardner = "square"
bounds_gardner = (0,6)


#################################
# Simulation 1 (gardner1)
#################################
# Also P1 in Lam Willcox Appendix

# TODO: Restructure main script to always give same dimension
# TODO: Rewrtie all with multiple constraints method
def cost_gardner1(x):
    xs = x[:,0]
    ys = x[:,1]

    f = np.cos(2*xs) * np.cos(ys) + np.sin(xs)

    return f

def constraint_gardner1(x):
    xs = x[:,0]
    ys = x[:,1]

    c = np.cos(xs)*np.cos(ys) - np.sin(xs)*np.sin(ys)

    return c <= -0.5 #TODO: Check paper for positive defenition of its contraint vs how its implemeted(orig 0.5)
    #? Think its actually a typo in the paper


# Stored functions
gardner1 = {"Bound Type" : boundtype_gardner,
            "Bounds" : bounds_gardner,
            "Cost Function (x)":  lambda x: cost_gardner1(x),
            "Constraint Function (z)": [lambda z: 1-constraint_gardner1(z)]}

#################################
# Simulation 2 (gardner2)
#################################

def cost_gardner2(x):
    xs = x[:,0]
    ys = x[:,1]

    f = np.sin(xs) + ys

    return f

def constraint_gardner2(x):
    xs = x[:,0]
    ys = x[:,1]

    c = np.sin(xs)*np.sin(ys)

    return c <= -0.95

gardner2 = {"Bound Type" : boundtype_gardner,
            "Bounds" : bounds_gardner,
            "Cost Function (x)":  lambda x: cost_gardner2(x),
            "Constraint Function (z)": [lambda z: 1 - constraint_gardner2(z)]}


if __name__ == "__main__":
    range_min,range_max = bounds_gardner[0], bounds_gardner[-1]
    point_per_axis_start = 5 #7 is original
    xx0=np.linspace(range_min,range_max,point_per_axis_start)[1:-1]

    print(xx0)

#################################
# (gramacy)
#################################
# P2 in Lam Willcox Appendix

# Bounds and such
boundtype_gramacy = "square"
bounds_gramacy = (0,1)

def cost_gramacy(x):
    xs = x[:,0]
    ys = x[:,1]

    f = xs + ys

    return f

def constraint1_gramacy(x):

    return c <= 0

def constraint2_gramacy(x):

    return c <= 0

constraint_list_gramacy = [
    lambda z: 1 - constraint1_gramacy(z),
    lambda z: 1 - constraint2_gramacy(z)
    ]

gardner2 = {"Bound Type" : boundtype_gramacy,
            "Bounds" : bounds_gramacy,
            "Cost Function (x)":  lambda x: cost_gramacy(x),
            "Constraint Functions (z)": constraint_list_gramacy}




#################################
# P3 (lamwillcox3)
#################################

#################################
# EXAMPLE FORMAT
#################################
"""
# Bounds and such
boundtype_gramacy = "square"
bounds_gramacy = (0,1)

# Cost function
def cost_XXXXX(x):
    return f

# Constraint function
def constraint_XXXXX(x):
    return c <= 0

gardner2 = {"Bound Type" : boundtype_XXXX,
            "Bounds" : bounds_XXXX,
            "Cost Function (x)":  lambda x: cost_XXXX(x),
            "Constraint Function (z)": lambda z: 1 - constraint_XXXX(z)}
"""