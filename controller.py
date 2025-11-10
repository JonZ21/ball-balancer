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
    if ball_position is None or s is None:
        return [0,0,0]
    
    xy_error = ball_position - s #Element wise numpy subtraction

    #Obtain 1D errors by projecting onto the motors axis
    #returns an array of errors projected to each axis u1, u2, u3 respectively
    errors= [np.dot(u1, xy_error), np.dot(u2, xy_error), np.dot(u3, xy_error)] 

    return errors

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd, min_motor_angle, max_motor_angle):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.min_output_angle = min_motor_angle  # degrees (0 = up)
        self.max_output_angle = max_motor_angle   # degrees (20 = down)
        self.neutral_angle = 10  # Neutral/resting position

    def update(self, error, dt=0.033): #default dt is ~30fps the inverse of that is the seconds
        # Proportional term
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.previous_error) / dt

        # Calculate output relative to neutral angle (10 degrees)
        output = self.neutral_angle + (P + I + D)
        
        # Clip to valid range [0, 20] where 0=up, 10=neutral, 20=down
        output = np.clip(output, self.min_output_angle, self.max_output_angle)
        
        self.previous_error = error
        
        return output
    
