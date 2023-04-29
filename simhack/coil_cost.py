
import tcd_utils
from simnibs.mesh_tools import read_msh
import sys
import numpy as np
try:
    coil = tcd_utils.read_tcd('H1.tcd')
    fn = 'ernie_noneck.msh'
except:
    coil = tcd_utils.read_tcd('simhack/H1.tcd')
    fn = 'simhack/ernie_noneck.msh'
    

mesh = read_msh(fn)
skin = mesh.crop_mesh(1005)



if __name__ == "__main__":
    
    #init pos
    affine=np.eye(4)
    affine[:3,3]=[-30,120,100]
    affine[0,0]=-1
    affine[1,1]=1
    affine[2,2]=-1
    constr_func, cost_func = tcd_utils.get_costs_sep(coil, skin, affine) #Fetching function to run
    
    in_string = sys.argv[1]
    in_string  = in_string[1:-1]
    elements = in_string.split(",")
    assert len(elements) == 6, f"Wrong ({len(elements)}) number of elements in input: {in_string}"
    tup_in = np.array([float(e) for e in elements])
    
    #cost_fun = tcd_utils.get_costf(coil, skin, affine)
    #print(constr_func(tup_in))
    print(cost_func((tup_in)))