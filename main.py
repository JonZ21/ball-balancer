# from serial import SerialPort
from controller import projected_errors, PIDcontroller
from serial_interface import SerialPort
from ball_tracking import BallDetector
import cv2
import json
import numpy as np

with open("config.json", "r") as f:
    config = json.load(f) #Open the camera config file. 

detect = BallDetector(config)
cap = cv2.VideoCapture(config['camera']['index'], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# Load and resize calibration data by 2 (divide by 2 for 320x240 frame)
center_point_px = tuple((np.array(config['camera']['center_point_px']) / 2).astype(int))
platform_points = [tuple((np.array(pt) / 2).astype(int)) for pt in config['calibration']['platform_points']]
u1 = np.array(config['calibration']['unit_vectors']['u1'])
u2 = np.array(config['calibration']['unit_vectors']['u2'])  
u3 = np.array(config['calibration']['unit_vectors']['u3'])
u_vectors = [u1, u2, u3]

# Create window for ball tracking GUI
cv2.namedWindow("Ball Tracking - Real-time Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ball Tracking - Real-time Detection", 800, 600)

print("[INFO] Ball tracking started. Press 'q' to quit.")
print("[INFO] Center point marked with white vertical line")
print("[INFO] Ball position shown with green circle")


# Placeholder to import the config code, obtain each unit vector for the axis
# import the centroid location
# setup for ball tracking as necessary

#establish connection to serial port
serial_port = SerialPort()
serial_port.connect_serial()

#Send neutral angles to the motors on startup
serial_port.send_servo_angles(10, 10, 10)

#initialize PID controllers
Kp = 0.15
Kd = -0.03
Ki = 0

#Using placeholder motor angles for now, validate for each motor.
min_motor_angle = 0
max_motor_angle = 20

motor1_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle, max_motor_angle)
motor2_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle +6, max_motor_angle +6)
motor3_pid = PIDcontroller(Kp, Kd, Ki, min_motor_angle, max_motor_angle)

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))

    if not ret:
        continue
    
    # Draw detection overlay with ball circle and center point
    overlay, found= detect.draw_detection(frame, center_point_px, platform_points, u_vectors, show_info=True)
    
    # Add additional information panel
    if found:
        cv2.putText(overlay, f"Ball Detected: YES", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, f"Ball Detected: NO", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the frame with overlays
    cv2.imshow("Ball Tracking - Real-time Detection", overlay)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ball tracking stopped.")
        break


    #Calculate the ball position
    found, x, y, radius =  detect.detect_ball(frame)#Placeholder, Kean's function should output a 2D numpy array

    # Convert to numpy arrays - only if ball is detected
    if found and x is not None and y is not None:
        # detect_ball is called on the resized (320x240) frame, so x,y are already in the
        # resized coordinate system — do NOT divide by 2 here.
        ball_position = np.array([x, y])
    else:
        ball_position = None

    # Center point (resized and cast to int earlier)
    center = np.array(center_point_px)
    
    print("center point px:", center_point_px[0], center_point_px[1])
    print(f"Ball Position: {ball_position}, Center: {center}")

    #Obtain and project the error onto each axis. All inputs are 2D numpy arrays
    error_array = projected_errors(u1, u2, u3, ball_position, center) #output array is: [axis 1, axis 2, axis 3]
    
    #run each PID controller
    motor1_command = motor1_pid.update(error_array[0])
    motor2_command = motor2_pid.update(error_array[1])
    motor3_command = motor3_pid.update(error_array[2])

    #Send code to each motor by converting the commands to a 3 byte array and sending over serial
    serial_port.send_servo_angles(motor1_command, motor2_command, motor3_command)

    # delay(10)
    
cap.release()
cv2.destroyAllWindows()
