import ball_tracking
import camera_calibration
from controller import projected_errors, PIDcontroller

#Placeholder to import the config code, obtain each unit vector for the axis
#import the centroid location
#setup for ball tracking as necessary

#initialize PID controllers
Kp = 1
Kd = 0
Ki = 0

#Using placeholder motor angles for now, validate for each motor.
min_motor_angle = 0
max_motor_angle = 20

motor1_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle, max_motor_angle)
motor2_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle, max_motor_angle)
motor3_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle, max_motor_angle)

while(True):
    
    #Calculate the ball position
    ball_position = None #Placeholder, Kean's function should output a 2D numpy array

    #Obtain and project the error onto each axis. All inputs are 2D numpy arrays
    error_array = projected_errors(u1, u2, u3, ball_position) #output array is: [axis 1, axis 2, axis 3]
    
    #run each PID controller
    motor1_command = motor1_pid.update(error_array(1))
    motor2_command = motor2_pid.update(error_array(2))
    motor3_command = motor3_pid.update(error_array(3))

    #Send code to each motor by converting the commands to a 3 byte array and sending over serial
    
