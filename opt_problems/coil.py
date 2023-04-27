
#################################
# EXAMPLE FORMAT
#################################
# Bounds and such
boundtype_coil = "different"
bounds_coil = =
import subprocess

# Cost function
def cost_coil(x):
    #Call subprocess for easier impl with pesc?
    string_out = ??? 
    out_pros_result = subprocess.run(["conda", "run", "-n", "simnibs", "python", 
                               ".\cost_coil", string_out],shell = True, capture_output=True)
    
    f = ###
    
    return f

"""
string_out = "["
for v in xx:
    string_out += str(v) + ","
    
"""

# Constraint function
def constraint1_coil(x):
    
    #Call subprocess for easier impl with pesc?
    out_pros_result = subprocess.run(["conda", "run", "-n", "simnibs", "python", 
                               ".\cost_coil"],shell = True, capture_output=True)
    c = ###
    return c <= 0

    
constraint_list_coil = [
    lambda z: 1 - constraint1_coil(z),
    ]

NAME = {"Bound Type" : boundtype_coil,
            "Bounds" : bounds_coil,
            "Cost Function (x)":  lambda x: cost_coil(x),
            "Constraint Functions (z)": constraint_list_coil}