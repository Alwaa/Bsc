
import tcd_utils
import numpy as np
from simnibs.mesh_tools import read_msh
import simnibs
import simnibs.simulation
import numpy as np

coil = tcd_utils.read_tcd('H1.tcd')


# only important for hacky implementation?
coil['deformationList'][0]['initial']=-10
coil['deformationList'][1]['initial']=-10
coil['deformationList'][2]['initial']=-10
coil['deformationList'][-1]['initial']=-5



fn = 'ernie_noneck.msh'

mesh = read_msh(fn)
skin = mesh.crop_mesh(1005)

#init pos
affine=np.eye(4)
affine[:3,3]=[-30,120,100]
affine[0,0]=-1
affine[1,1]=1
affine[2,2]=-1

#vizualize position
# tcd_utils.vizCoil(coil, affine, (0,0,0,0,0,0), skin)

constr_func, cost_func = tcd_utils.get_costs_sep(coil, skin, affine)

print(tcd_utils.print_bounds(coil))
print(constr_func((0,0,0,0,0,0)))
print(cost_func((0,0,0,0,0,0)))
print(constr_func((-10., -26.7, -10.,    0.,    0.,  -25.1)))
print(cost_func((-10., -26.7, -10.,    0.,    0.,  -25.1)))
print(constr_func((-10., -26.7, -10.,    0.,    0.,  -47.3)))
print(cost_func((-10., -26.7, -10.,    0.,    0.,  -47.3)))

res1 = tcd_utils.optimize_position(coil, skin, affine) #use global solver + refinement via lbfgs

p=res1.x
# p=(0,0,0,0,-20,0)
print("---",p,"---")
print(constr_func(p))
print(cost_func((p)))
tcd_utils.vizCoil(coil, affine, p, skin)


