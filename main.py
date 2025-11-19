# from serial import SerialPort
from controller import projected_errors, PIDcontroller
from serial_interface import SerialPort
from ball_tracking import BallDetector
from tuning_j import start_trial, log_sample, finish_and_score
from nm_tuner import start_nm_tuning
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time

with open("config.json", "r") as f:
    config = json.load(f) #Open the camera config file. 

detect = BallDetector("config.json")
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
print("[INFO] Use PID Tuning GUI to adjust Kp, Ki, Kd values and click 'Send' to update")
print("[INFO] Center point marked with red dot")
print("[INFO] Ball position shown with green circle")
print("[INFO] Current PID values displayed on video feed")


# Placeholder to import the config code, obtain each unit vector for the axis
# import the centroid location
# setup for ball tracking as necessary

#establish connection to serial port
serial_port = SerialPort()
serial_port.connect_serial()

#Send neutral angles to the motors on startup
serial_port.send_servo_angles(10, 10, 10)

#initialize PID controllers
# [NM] BEST  Kp=0.1899  Ki=0.0000  Kd=0.1075   ->   J=69.283

#Kp = 0.16
#Ki = 0.007
#Kd = 0.09
Kp = 0.1953
Ki = 0.0010
Kd = 0.1372
#Kp=0.1956  Ki=0.0000  Kd=0.1150   ->   J=525.398
#BEST  Kp=0.2015  Ki=0.0011  Kd=0.1438   ->   J=610.338
#Starting trial with Kp=0.2022, Ki=0.0011, Kd=0.1232.
#tarting trial with Kp=0.2056, Ki=0.0012, Kd=0.1283 this converged
#BEST  Kp=0.1953  Ki=0.0010  Kd=0.1372   ->   J=623.524

# PID tuning ranges for sliders
Kp_max = 1
Ki_max = 0.8
Kd_max = 1
slider_resolution = 1000  # Higher resolution for finer tuning

# Deadzone settings
deadzone_radius = 13.0  # pixels
deadzone_max = 20.0  # maximum deadzone radius

#Using placeholder motor angles for now, validate for each motor.
min_motor_angle = 0
max_motor_angle = 20

motor1_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle, max_motor_angle)
motor2_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle +6, max_motor_angle +6)
motor3_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle, max_motor_angle)

# Last valid camera error (used if detection drops out briefly)
last_xy_error = np.array([0.0, 0.0], dtype=float)

# Global variables for setpoint
use_custom_setpoint = False
custom_setpoint = center_point_px  # Initialize to center
setpoint_lock = threading.Lock()

# Global variables for GUI
pid_gains_lock = threading.Lock()
current_kp = Kp
current_ki = Ki
current_kd = Kd
current_deadzone = deadzone_radius

# PID Tuning GUI Class
class PIDTuningGUI:
    def __init__(self, root, motor1_pid, motor2_pid, motor3_pid):
        self.root = root
        self.motor1_pid = motor1_pid
        self.motor2_pid = motor2_pid
        self.motor3_pid = motor3_pid

        self.root.title("PID Tuning")
        # Larger window for better visibility
        self.root.geometry("550x1000+850+100")  # Position window beside video feed (x=850, y=100)
        self.root.resizable(True, True)

        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Make columns expand so widgets like the Send button stretch to full width
        main_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="PID Gain Tuning", font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 30))

        # Kp slider
        self.kp_label = ttk.Label(main_frame, text=f"Kp: {Kp:.4f}", font=("Arial", 16))
        self.kp_label.grid(row=1, column=0, columnspan=3, pady=10)
        self.kp_scale = ttk.Scale(main_frame, from_=0, to=Kp_max, length=450,
                                   orient=tk.HORIZONTAL, command=self.update_kp_label)
        self.kp_scale.set(Kp)
        self.kp_scale.grid(row=2, column=0, columnspan=3, pady=10)

        # Ki slider
        self.ki_label = ttk.Label(main_frame, text=f"Ki: {Ki:.4f}", font=("Arial", 16))
        self.ki_label.grid(row=3, column=0, columnspan=3, pady=10)
        self.ki_scale = ttk.Scale(main_frame, from_=0, to=Ki_max, length=450,
                                   orient=tk.HORIZONTAL, command=self.update_ki_label)
        self.ki_scale.set(Ki)
        self.ki_scale.grid(row=4, column=0, columnspan=3, pady=10)

        # Kd slider
        self.kd_label = ttk.Label(main_frame, text=f"Kd: {Kd:.4f}", font=("Arial", 16))
        self.kd_label.grid(row=5, column=0, columnspan=3, pady=10)
        self.kd_scale = ttk.Scale(main_frame, from_=0, to=Kd_max, length=450,
                                   orient=tk.HORIZONTAL, command=self.update_kd_label)
        self.kd_scale.set(Kd)
        self.kd_scale.grid(row=6, column=0, columnspan=3, pady=10)

        # Deadzone slider
        self.deadzone_label = ttk.Label(main_frame, text=f"Deadzone: {deadzone_radius:.1f} px", font=("Arial", 16))
        self.deadzone_label.grid(row=7, column=0, columnspan=3, pady=10)
        self.deadzone_scale = ttk.Scale(main_frame, from_=0, to=deadzone_max, length=450,
                                         orient=tk.HORIZONTAL, command=self.update_deadzone_label)
        self.deadzone_scale.set(deadzone_radius)
        self.deadzone_scale.grid(row=8, column=0, columnspan=3, pady=10)

        # Current values display
        current_frame = ttk.LabelFrame(main_frame, text="Current Values", padding="15")
        current_frame.grid(row=9, column=0, columnspan=3, pady=25, sticky=(tk.W, tk.E))

        self.current_kp_label = ttk.Label(current_frame, text=f"Kp: {Kp:.4f}", font=("Arial", 14))
        self.current_kp_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.current_ki_label = ttk.Label(current_frame, text=f"Ki: {Ki:.4f}", font=("Arial", 14))
        self.current_ki_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.current_kd_label = ttk.Label(current_frame, text=f"Kd: {Kd:.4f}", font=("Arial", 14))
        self.current_kd_label.grid(row=2, column=0, sticky=tk.W, pady=5)

        self.current_deadzone_label = ttk.Label(current_frame, text=f"Deadzone: {deadzone_radius:.1f} px", font=("Arial", 14))
        self.current_deadzone_label.grid(row=3, column=0, sticky=tk.W, pady=5)

        # Send button
        send_button = ttk.Button(main_frame, text="Send", command=self.send_pid_values)
        # Make the Send button stretch horizontally and ensure it's visible
        send_button.grid(row=10, column=0, columnspan=3, pady=15, sticky=(tk.E, tk.W))
        # Configure button style to make it bigger
        style = ttk.Style()
        style.configure('Big.TButton', font=('Arial', 14))
        send_button.configure(style='Big.TButton')

        # Reset Integral button
        reset_button = ttk.Button(main_frame, text="Reset Integral Error", command=self.reset_integral)
        reset_button.grid(row=11, column=0, columnspan=3, pady=10, sticky=(tk.E, tk.W))
        reset_button.configure(style='Big.TButton')

        # Setpoint controls frame
        setpoint_frame = ttk.LabelFrame(main_frame, text="Setpoint Control", padding="15")
        setpoint_frame.grid(row=12, column=0, columnspan=3, pady=25, sticky=(tk.W, tk.E))

        # Current setpoint display
        self.setpoint_status_label = ttk.Label(setpoint_frame, text="Using: Center Point", font=("Arial", 14, "bold"))
        self.setpoint_status_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # X coordinate input
        ttk.Label(setpoint_frame, text="X:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.E, padx=(0, 5))
        self.setpoint_x_entry = ttk.Entry(setpoint_frame, width=10, font=("Arial", 12))
        self.setpoint_x_entry.insert(0, str(center_point_px[0]))
        self.setpoint_x_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Y coordinate input
        ttk.Label(setpoint_frame, text="Y:", font=("Arial", 12)).grid(row=2, column=0, sticky=tk.E, padx=(0, 5))
        self.setpoint_y_entry = ttk.Entry(setpoint_frame, width=10, font=("Arial", 12))
        self.setpoint_y_entry.insert(0, str(center_point_px[1]))
        self.setpoint_y_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Set Setpoint button
        set_setpoint_button = ttk.Button(setpoint_frame, text="Set Custom Setpoint", command=self.set_custom_setpoint)
        set_setpoint_button.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.E, tk.W))
        set_setpoint_button.configure(style='Big.TButton')

        # Use Center button
        use_center_button = ttk.Button(setpoint_frame, text="Use Center Point", command=self.use_center_setpoint)
        use_center_button.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.E, tk.W))
        use_center_button.configure(style='Big.TButton')

        # Update current values periodically
        self.update_current_values()
        self.update_setpoint_display()
    
    def update_kp_label(self, value):
        """Update Kp label as slider moves."""
        kp_val = float(value)
        self.kp_label.config(text=f"Kp: {kp_val:.4f}")
    
    def update_ki_label(self, value):
        """Update Ki label as slider moves."""
        ki_val = float(value)
        self.ki_label.config(text=f"Ki: {ki_val:.4f}")
    
    def update_kd_label(self, value):
        """Update Kd label as slider moves."""
        kd_val = float(value)
        self.kd_label.config(text=f"Kd: {kd_val:.4f}")

    def update_deadzone_label(self, value):
        """Update Deadzone label as slider moves."""
        deadzone_val = float(value)
        self.deadzone_label.config(text=f"Deadzone: {deadzone_val:.1f} px")

    def send_pid_values(self):
        """Update PID controllers with slider values."""
        global current_kp, current_ki, current_kd, current_deadzone

        kp_val = self.kp_scale.get()
        ki_val = self.ki_scale.get()
        kd_val = self.kd_scale.get()
        deadzone_val = self.deadzone_scale.get()

        with pid_gains_lock:
            current_kp = kp_val
            current_ki = ki_val
            current_kd = kd_val
            current_deadzone = deadzone_val

        # Update all PID controllers
        self.motor1_pid.update_gains(Kp=kp_val, Ki=ki_val, Kd=kd_val)
        self.motor2_pid.update_gains(Kp=kp_val, Ki=ki_val, Kd=kd_val)
        self.motor3_pid.update_gains(Kp=kp_val, Ki=ki_val, Kd=kd_val)

        print(f"[PID] Updated gains - Kp: {kp_val:.4f}, Ki: {ki_val:.4f}, Kd: {kd_val:.4f}, Deadzone: {deadzone_val:.1f} px")

    def reset_integral(self):
        """Reset integral error for all PID controllers."""
        self.motor1_pid.reset_integral()
        self.motor2_pid.reset_integral()
        self.motor3_pid.reset_integral()
        print("[PID] Integral error reset for all controllers")

    def update_current_values(self):
        """Update displayed current values."""
        global current_kp, current_ki, current_kd, current_deadzone

        with pid_gains_lock:
            kp = current_kp
            ki = current_ki
            kd = current_kd
            deadzone = current_deadzone

        self.current_kp_label.config(text=f"Kp: {kp:.4f}")
        self.current_ki_label.config(text=f"Ki: {ki:.4f}")
        self.current_kd_label.config(text=f"Kd: {kd:.4f}")
        self.current_deadzone_label.config(text=f"Deadzone: {deadzone:.1f} px")

        # Schedule next update
        self.root.after(100, self.update_current_values)

    def sync_sliders_from_globals(self):
        """Sync GUI sliders with global gain values (called by NM tuner)."""
        global current_kp, current_ki, current_kd
        with pid_gains_lock:
            self.kp_scale.set(current_kp)
            self.ki_scale.set(current_ki)
            self.kd_scale.set(current_kd)
        print("[GUI] Sliders synced with optimized gains")

    def set_custom_setpoint(self):
        """Set a custom setpoint from the entry fields."""
        global use_custom_setpoint, custom_setpoint
        try:
            x = int(self.setpoint_x_entry.get())
            y = int(self.setpoint_y_entry.get())

            with setpoint_lock:
                custom_setpoint = (x, y)
                use_custom_setpoint = True

            print(f"[SETPOINT] Custom setpoint set to ({x}, {y})")
        except ValueError:
            print("[SETPOINT] Error: Invalid X or Y value. Please enter integers.")

    def use_center_setpoint(self):
        """Switch back to using the center point."""
        global use_custom_setpoint

        with setpoint_lock:
            use_custom_setpoint = False

        # Reset entry fields to center
        self.setpoint_x_entry.delete(0, tk.END)
        self.setpoint_x_entry.insert(0, str(center_point_px[0]))
        self.setpoint_y_entry.delete(0, tk.END)
        self.setpoint_y_entry.insert(0, str(center_point_px[1]))

        print("[SETPOINT] Using center point")

    def update_setpoint_display(self):
        """Update the setpoint status display."""
        global use_custom_setpoint, custom_setpoint

        with setpoint_lock:
            if use_custom_setpoint:
                self.setpoint_status_label.config(text=f"Using: Custom ({custom_setpoint[0]}, {custom_setpoint[1]})")
            else:
                self.setpoint_status_label.config(text="Using: Center Point")

        # Schedule next update
        self.root.after(100, self.update_setpoint_display)

# Global reference to GUI app for NM tuner to update sliders
gui_app = None

def run_gui():
    """Run the PID tuning GUI in a separate thread."""
    global gui_app
    root = tk.Tk()
    gui_app = PIDTuningGUI(root, motor1_pid, motor2_pid, motor3_pid)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

# Start GUI in a separate thread
gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

# Small delay to allow GUI to initialize
time.sleep(0.5)
count = 0 
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
    
    # Display current PID values on overlay
    with pid_gains_lock:
        display_kp = current_kp
        display_ki = current_ki
        display_kd = current_kd
        display_deadzone = current_deadzone

    cv2.putText(overlay, f"Kp: {display_kp:.4f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"Ki: {display_ki:.4f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"Kd: {display_kd:.4f}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"Deadzone: {display_deadzone:.1f} px", (10, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw deadzone circle overlay
    cv2.circle(overlay, center_point_px, int(display_deadzone), (255, 0, 255), 2)

    # Draw current setpoint marker
    with setpoint_lock:
        if use_custom_setpoint:
            setpoint_to_draw = custom_setpoint
            # Draw custom setpoint as a blue crosshair
            cross_size = 10
            cv2.line(overlay, (setpoint_to_draw[0] - cross_size, setpoint_to_draw[1]),
                    (setpoint_to_draw[0] + cross_size, setpoint_to_draw[1]), (255, 255, 0), 2)
            cv2.line(overlay, (setpoint_to_draw[0], setpoint_to_draw[1] - cross_size),
                    (setpoint_to_draw[0], setpoint_to_draw[1] + cross_size), (255, 255, 0), 2)
            cv2.circle(overlay, setpoint_to_draw, 5, (255, 255, 0), -1)
            cv2.putText(overlay, "SETPOINT", (setpoint_to_draw[0] - 35, setpoint_to_draw[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Display the frame with overlays
    # Display the frame with overlays
    cv2.imshow("Ball Tracking - Real-time Detection", overlay)

    # ---- Single keyboard handler for the whole app (runs each frame) ----
    # OpenCV processes GUI + keyboard events only inside waitKey/pollKey.
    key = cv2.waitKey(1) & 0xFF

    # Quit the program
    if key == ord('q'):
        print("[INFO] Ball tracking stopped.")
        break
    
    elif key == ord('o'):
    # use current GUI slider values as initial guess (nice UX)
        with pid_gains_lock:
            x0 = (current_kp, current_ki, current_kd)
        print("[NM] Starting Nelder-Mead auto-tuner...")
        print("[NM] This will run multiple 10s trials. Watch the console for progress.")
        
        # Callback to update global variables
        def update_globals(kp, ki, kd):
            # Update all PID controllers
            print("Setting gains to: ")
            print("Kp:", kp, "Ki:", ki, "Kd:", kd)
            PIDTuningGUI.motor1_pid.update_gains(Kp=kp, Ki=ki, Kd=kd)
            PIDTuningGUI.motor2_pid.update_gains(Kp=kp, Ki=ki, Kd=kd)
            PIDTuningGUI.motor3_pid.update_gains(Kp=kp, Ki=ki, Kd=kd)

        # Callback to sync GUI sliders
        def sync_gui():
            if gui_app is not None:
                gui_app.root.after(0, gui_app.sync_sliders_from_globals)
        
        start_nm_tuning(
            motor_pids=(motor1_pid, motor2_pid, motor3_pid),
            trial_sec= 17.0,
            w1= 2, w2=0.8, w3=3.5, pctl=95,
            x0=x0,
            scale=0.4,
            max_iter= 14,
            gains_lock=pid_gains_lock,
            update_globals_fn=update_globals,
            sync_gui_fn=sync_gui
        )
        print("[NM] Tuner started in background. System will continue running.")

    # Start a ~10 s scoring trial:
    # - Reset integrators so trials are comparable (no old integral windup)
    # - Start the scorer's buffers
    elif key == ord('s'):
        motor1_pid.reset_integral()
        motor2_pid.reset_integral()
        motor3_pid.reset_integral()
        start_trial()
        print("[TUNE] Trial started (~10 s). Press 'e' to end & score.")

    # End trial and compute J = w1*IAE + w2*P95 on raw r[k] (no filtering).
    # Prints both the overall score and a breakdown useful for debugging.
    elif key == ord('e'):
        J, parts = finish_and_score(w1=1.0, w2=0.7, pctl=95)
        if J is None:
            print("[TUNE] Not enough samples to score.")
        else:
            print(f"[TUNE] J={J:.3f}  IAE={parts['IAE']:.2f}  "
                f"P95={parts['P95']:.2f} OSC={parts['OSC']:.2f}  N={parts['N']}  dt≈{parts['dt']:.3f}s")


    #Calculate the ball position
    found, x, y, radius =  detect.detect_ball(frame)#Placeholder, Kean's function should output a 2D numpy array

    # Convert to numpy arrays - only if ball is detected
    if found and x is not None and y is not None:
        # detect_ball is called on the resized (320x240) frame, so x,y are already in the
        # resized coordinate system — do NOT divide by 2 here.
        ball_position = np.array([x, y])
    else:
        ball_position = None

    # Get the current setpoint (either center or custom)
    with setpoint_lock:
        if use_custom_setpoint:
            current_setpoint = np.array(custom_setpoint)
        else:
            current_setpoint = np.array(center_point_px)

    # ---- Build raw 2-D camera error and log it for scoring ----
    if ball_position is not None:
        xy_error = ball_position - current_setpoint          # shape (2,)
        last_xy_error = xy_error
    else:
        xy_error = last_xy_error                   # keep logging during short dropouts

    log_sample(xy_error)                           # no-op unless a trial is active

    # print("center point px:", center_point_px[0], center_point_px[1])
    # print(f"Ball Position: {ball_position}, Setpoint: {current_setpoint}")

    # Get current deadzone value
    with pid_gains_lock:
        deadzone_value = current_deadzone

    #Obtain and project the error onto each axis. All inputs are 2D numpy arrays
    error_array, count  = projected_errors(u1, u2, u3, ball_position, current_setpoint, deadzone_value, count) #output array is: [axis 1, axis 2, axis 3]

    #run each PID controller
    motor1_command = motor1_pid.update(error_array[0])
    motor2_command = motor2_pid.update(error_array[1])
    motor3_command = motor3_pid.update(error_array[2])

    #Send code to each motor by converting the commands to a 3 byte array and sending over serial
    serial_port.send_servo_angles(motor1_command, motor2_command, motor3_command)

    # delay(10)
    
cap.release()
cv2.destroyAllWindows()
