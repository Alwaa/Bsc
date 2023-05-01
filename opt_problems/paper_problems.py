import numpy as np

# http://proceedings.mlr.press/v32/gardner14.pdf
# Originally from Jacob R Gardner, Matt J Kusner, Zhixiang Eddie Xu, Kilian Q Weinberger, and John P Cunningham. Bayesian optimization with inequality constraints.

#--------------------------------#
# Bounds and such
#--------------------------------#
e = 0 #TODO. Investigate why lower corner gets plotted as feasable
boundtype_gardner = "square"
bounds_gardner = (0+e,6-e,0+e,6-e)

#################################
# Simulation 1 (gardner1)
#################################
# Also P1 in Lam Willcox Appendix

# TODO: Restructure main script to always give same dimension
# TODO: Rewrtie all with multiple constraints method
def cost_gardner1(x):
    xx = x[None] if x.ndim == 1 else x # In case of single value input
    xs = xx[:,0]
    ys = xx[:,1]

    f = np.cos(2*xs) * np.cos(ys) + np.sin(xs)

    return f

def constraint_gardner1(x):
    xx = x[None] if x.ndim == 1 else x # In case of single value input
    xs = xx[:,0]
    ys = xx[:,1]

    c = np.cos(xs)*np.cos(ys) - np.sin(xs)*np.sin(ys) + 0.5

    return c <= 0 #TODO: Check paper for positive defenition of its contraint vs how its implemeted(orig 0.5)
    #? Think its actually a typo in the paper


# Stored functions
gardner1 = {"Bound Type" : boundtype_gardner,
            "Bounds" : bounds_gardner,
            "Cost Function (x)":  lambda x: cost_gardner1(x),
            "Constraint Functions (z)": [lambda z: 1-constraint_gardner1(z)]}

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

    c = np.sin(xs)*np.sin(ys) + 0.95

    return c <= 0

gardner2 = {"Bound Type" : boundtype_gardner,
            "Bounds" : bounds_gardner,
            "Cost Function (x)":  lambda x: cost_gardner2(x),
            "Constraint Functions (z)": [lambda z: 1 - constraint_gardner2(z)]}


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
bounds_gramacy = (0,1,0,1)

def cost_gramacy(x):
    xs = x[:,0]
    ys = x[:,1]

    f = xs + ys

    return f

def constraint1_gramacy(x):
    xs = x[:,0]
    ys = x[:,1]
    c = 0.5*np.sin(2*np.pi*(2*ys - xs**2)) - xs - 2*ys + 1.5
    return c <= 0

def constraint2_gramacy(x):
    xs = x[:,0]
    ys = x[:,1]
    
    c = xs**2 + ys**2 -1.5
    
    return c <= 0

constraint_list_gramacy = [
    lambda z: 1 - constraint1_gramacy(z),
    lambda z: 1 - constraint2_gramacy(z)
    ]

gramacy = {"Bound Type" : boundtype_gramacy,
            "Bounds" : bounds_gramacy,
            "Cost Function (x)":  lambda x: cost_gramacy(x),
            "Constraint Functions (z)": constraint_list_gramacy}




#################################
# P3 (lamwillcox3)
#################################
# Bounds and such
boundtype_lw3 = "square"
bounds_lw3 = (-5,5)*4

# Cost function
def cost_lw3(x):
    
    f = 0.5*np.sum(x**4 - 16*(x**2) + 5*x,axis = 1)
    
    return f

# Constraint function
def constraint1_lw3(x):
    
    c = -0.5 + np.sin(x[:,0] + 2*x[:,1]) -np.cos(x[:,2]*np.cos(x[:,3]))
    
    return c <= 0

    
    
constraint_list_lw3 = [
    lambda z: 1 - constraint1_lw3(z)
    ]

lamwillcox3 = {"Bound Type" : boundtype_lw3,
            "Bounds" : bounds_lw3,
            "Cost Function (x)":  lambda x: cost_lw3(x),
            "Constraint Functions (z)": constraint_list_lw3}
#################################
# EXAMPLE FORMAT
#################################
"""
# Bounds and such
boundtype_XXXX = "square"
bounds_XXXX = (0,1)

# Cost function
def cost_XXXXX(x):
    return f

# Constraint function
def constraint1_XXXXX(x):
    return c <= 0

def constraint2_XXXXX(x):
    return c <= 0
    
    
constraint_list_XXXX = [
    lambda z: 1 - constraint1_XXXX(z),
    lambda z: 1 - constraint2_XXXX(z)
    ]

NAME = {"Bound Type" : boundtype_XXXX,
            "Bounds" : bounds_XXXX,
            "Cost Function (x)":  lambda x: cost_XXXX(x),
            "Constraint Functions (z)": constraint_list_XXXX}
"""