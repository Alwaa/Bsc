import numpy as np

# http://proceedings.mlr.press/v32/gardner14.pdf
# Originally from Jacob R Gardner, Matt J Kusner, Zhixiang Eddie Xu, Kilian Q Weinberger, and John P Cunningham. Bayesian optimization with inequality constraints.

#--------------------------------#
# Bounds and such
#--------------------------------#

xin = np.linspace(0, 6, 301)
xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T

#################################
# Simulation 1 (gardner1)
#################################
def cost_gardner1(x):
    #assert x.shape[1] == 2, "-"*24 + f"\n Expected dim 2 input, got {x.shape[1]}"
    if x.ndim == 1:
        #x = x[None]
        xs, ys = x[0], x[1]
    else:
        xs = x[:,0]
        ys = x[:,1]

    f = np.cos(2*xs) * np.cos(ys) + np.sin(xs)

    return f

def constraint_gardner1(x):
    if x.ndim == 1:
        x = x[None]

    xs = x[:,0]
    ys = x[:,1]

    c = np.cos(xs)*np.cos(ys) - np.sin(xs)*np.sin(ys)

    return c <= 0.5


# Stored functions
gardner1 = {"X in" : xin,
            "XY" : xy,
            "Cost Function filled in" : cost_gardner1(xy), 
            "Constraint Function filled in" : constraint_gardner1(xy),
            "Cost Function (x)":  lambda x: cost_gardner1(x),
            "Constraint Function (z)": lambda z: 1-constraint_gardner1(z)}

#################################
# Simulation 2 (gardner2)
#################################

def cost_gardner2(x):
    #assert x.shape[1] == 2, "-"*24 + f"\n Expected dim 2 input, got {x.shape[1]}"
    if x.ndim == 1:
        #x = x[None]
        xs, ys = x[0], x[1]
    else:
        xs = x[:,0]
        ys = x[:,1]

    f = np.sin(xs) + ys

    return f

def constraint_gardner2(x):
    if x.ndim == 1:
        x = x[None]

    xs = x[:,0]
    ys = x[:,1]

    c = np.sin(xs)*np.sin(ys)

    return c <= -0.95

gardner2 = {"X in" : xin,
            "XY" : xy,
            "Cost Function filled in" : cost_gardner2(xy), 
            "Constraint Function filled in" : constraint_gardner2(xy),
            "Cost Function (x)":  lambda x: cost_gardner2(x),
            "Constraint Function (z)": lambda z: 1-constraint_gardner2(z)}


range_min,range_max = xin[0], xin[-1]
point_per_axis_start = 20 #7 is original
xx0=np.linspace(range_min,range_max,point_per_axis_start)[1:-1]

print(xx0)