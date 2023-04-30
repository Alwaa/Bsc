
#################################
# EXAMPLE FORMAT
#################################
# Bounds and such

import subprocess
import numpy as np
boundtype_coil = "different"
bounds_coil = (-25, 25, -25, 25, -25, 25, -20, 20, -50, 50, -50, 50)

# Cost function
def cost_coil(x):
    if len(np.shape(x)) == 1:
        xx = np.array([x])
    else:
        xx = x
    #Call subprocess for easier impl with pesc?
    sampl_num =len(xx)
    out = np.full(sampl_num,0.)
    for i in range(sampl_num):
        xi = xx[i]
        vals = []
        for val in xi:
            vals.append(str(val))
        string_out = "[" + ",".join(vals) + "]" 
        out_result = subprocess.check_output(["conda", "run", "-n", "simnibs", "python", 
                                "simhack\coil_cost.py", string_out])
        out[i] = float(out_result.split(sep=b"\r\r\n")[0])
    f = out
    
    return f


# Constraint function
def constraint1_coil(x):
    
    if len(np.shape(x)) == 1:
        xx = np.array([x])
    else:
        xx = x
    #Call subprocess for easier impl with pesc?
    sampl_num =len(xx)
    out = np.full(sampl_num,0.)
    for i in range(sampl_num):
        xi = xx[i]
        vals = []
        for val in xi:
            vals.append(str(val))
        string_out = "[" + ",".join(vals) + "]" 
        out_result = subprocess.check_output(["conda", "run", "-n", "simnibs", "python", 
                                "simhack\coil_constraint.py", string_out])
        out[i] = (out_result.split(sep=b"\r\r\n")[0] == b'True') ## Should prob do it better

    c = out

    return c <= 0

    
constraint_list_coil = [
    lambda z: 1 - constraint1_coil(z),
    ]

coil = {"Bound Type" : boundtype_coil,
            "Bounds" : bounds_coil,
            "Cost Function (x)":  lambda x: cost_coil(x),
            "Constraint Functions (z)": constraint_list_coil}

if __name__ == "__main__":
    from time import time
    tst = np.array([[0.,    0.,    0.,     0.,    0.,     0.],
                    [-10., -26.7, -10.,    0.,    0.,  -25.1],
                    [-10., -26.7, -10.,    0.,    0.,  -47.3],])

    print(tst)
    start_t = time()
    print(cost_coil(tst))
    cost_time = time()
    print(1-constraint1_coil(tst))
    constr_time = time()
    print(int(cost_time - start_t), int(constr_time-cost_time)) #TODO: Check if I can save time by loading just one func at a time

    