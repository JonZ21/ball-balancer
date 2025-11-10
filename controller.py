import numpy as np

def project_error (u1, u2, u3, ball_position, s):
    """Arguments: 
    u1, u2, u3 are the unit vectors (magnitude of 1)
    for each motor's axis on the stewart platform. They are an (x,y) list
    the ball position is input in the global camera x and y coordanates
    s is the centroid location of the stewart platform in the x and y
    ALL OF THESE ARE EXPECTED AS NUMPY ARRAYS

    Outputs:
    This function will output 3 errors, one for each motor's axis.
    """
    #The 2D error vector:
    xy_error = s-ball_position #Element wise numpy subtraction

    #Obtain 1D errors by projecting onto the motors axis
    #returns an array of errors projected to each axis u1, u2, u3 respectively
    errors= [np.dot(u1, xy_error), np.dot(u2, xy_error), np.dot(u3, xy_error)] 

    return errors