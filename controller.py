import numpy as np

def projected_errors (u1, u2, u3, ball_position, s):
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

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.min_output_angle = 0  # degrees
        self.max_output_angle = 20   # degrees

    def update(self, error, dt=0.033): #default dt is ~30fps the inverse of that is the seconds
        # Proportional term
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.previous_error) / dt

        output = P + I + D
        output = np.clip(output, self.min_output_angle, self.max_output_angle)
        
        return output
    
