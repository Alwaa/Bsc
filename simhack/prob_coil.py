#################################
# EXAMPLE FORMAT
#################################
# Bounds and such


import numpy as np
boundtype_coil = "different"
bounds_coil = (-25, 25, -25, 25, -25, 25, -20, 20, -50, 50, -50, 50)


import simhack.tcd_utils as tcd_utils
from simhack.simnibs.mesh_tools import read_msh
import numpy as np
try:
    coil = tcd_utils.read_tcd('H1.tcd')
    fn = 'ernie_noneck.msh'
except:
    coil = tcd_utils.read_tcd('simhack/H1.tcd')
    fn = 'simhack/ernie_noneck.msh'

mesh = read_msh(fn)
skin = mesh.crop_mesh(1005)
  
#init pos
affine=np.eye(4)
affine[:3,3]=[-30,120,100]
affine[0,0]=-1
affine[1,1]=1
affine[2,2]=-1
constr_func, cost_func = tcd_utils.get_costs_sep(coil, skin, affine) #Fetching function to run
# Cost function
def cost_coil_pure(x):
    if len(np.shape(x)) == 1:
        xx = np.array([x])
    else:
        xx = x
    #Call subprocess for easier impl with pesc?
    sampl_num =len(xx)
    out = np.full(sampl_num,0.)
    for i in range(sampl_num):
        xi = xx[i]
        out[i] = cost_func((xi))
    f = out
    
    return f


# Constraint function
def constraint1_coil_pure(x):
    
    if len(np.shape(x)) == 1:
        xx = np.array([x])
    else:
        xx = x
    #Call subprocess for easier impl with pesc?
    sampl_num =len(xx)
    out = np.full(sampl_num,0.)
    for i in range(sampl_num):
        xi = xx[i]
        out[i] = constr_func(xi)

    c = out

    return c <= 0

    
constraint_list_coil_pure = [
    lambda z: 1 - constraint1_coil_pure(z),
    ]

coil_pure = {"Bound Type" : boundtype_coil,
            "Bounds" : bounds_coil,
            "Cost Function (x)":  lambda x: cost_coil_pure(x),
            "Constraint Functions (z)": constraint_list_coil_pure}
