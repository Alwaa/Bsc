
#################################
# EXAMPLE FORMAT
#################################
# Bounds and such
boundtype_coil = "different"
bounds_coil = =

# Cost function
def cost_coil(x):
    
    f = #Call subprocess for easier impl with pesc?
    
    return f

# Constraint function
def constraint1_coil(x):
    
    c = #Call subprocess for easier impl with pesc?
    
    return c <= 0

    
constraint_list_coil = [
    lambda z: 1 - constraint1_coil(z),
    ]

NAME = {"Bound Type" : boundtype_coil,
            "Bounds" : bounds_coil,
            "Cost Function (x)":  lambda x: cost_coil(x),
            "Constraint Functions (z)": constraint_list_coil}