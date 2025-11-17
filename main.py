# from serial import SerialPort
from controller import projected_errors, PIDcontroller
from serial_interface import SerialPort
from ball_tracking import BallDetector
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
Kp = 0.1950
Kd = 0.0872
Ki = 0.0200

# from autotune after 40 
# Kp=0.0801  
# Ki=0.1094  
# Kd=0.1787  

# PID tuning ranges for sliders
Kp_max = 1
Ki_max = 1
Kd_max = 1
slider_resolution = 1000  # Higher resolution for finer tuning

# Deadzone settings
deadzone_radius = 10.0  # pixels
deadzone_max = 20.0  # maximum deadzone radius

#Using placeholder motor angles for now, validate for each motor.
min_motor_angle = 0
max_motor_angle = 20

motor1_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle, max_motor_angle)  #PID not PDI
motor2_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle +6, max_motor_angle +6)
motor3_pid = PIDcontroller(Kp, Ki, Kd, min_motor_angle, max_motor_angle)

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
        self.root.geometry("550x800+850+100")  # Position window beside video feed (x=850, y=100)
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

        # Update current values periodically
        self.update_current_values()
    
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

def run_gui():
    """Run the PID tuning GUI in a separate thread."""
    root = tk.Tk()
    app = PIDTuningGUI(root, motor1_pid, motor2_pid, motor3_pid)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

#This is a gui class that will live update. 

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TrackingPlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Position Tracking")
        self.root.geometry("900x800+1450+100")
        self.root.resizable(True, True)

        # Data storage
        self.time_history = []
        self.x_history = []
        self.y_history = []
        self.xy_mag_history = []
        self.start_time = time.time()

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax_x = self.fig.add_subplot(311)
        self.ax_y = self.fig.add_subplot(312)
        self.ax_xy = self.fig.add_subplot(313)

        self.ax_x.set_title("X Position")
        self.ax_y.set_title("Y Position")
        self.ax_xy.set_title("XY Magnitude")

        for ax in [self.ax_x, self.ax_y, self.ax_xy]:
            ax.grid(True)

        # Canvas in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)

        save_btn = ttk.Button(btn_frame, text="Save + Clear", command=self.save_and_clear)
        save_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_plots)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Schedule updates
        self.update_plots()

    def add_data(self, x, y):
        """Called from main loop. Safe even if x,y = None."""
        if x is None or y is None:
            return
        
        t = time.time() - self.start_time
        mag = np.sqrt(x*x + y*y)

        self.time_history.append(t)
        self.x_history.append(x)
        self.y_history.append(y)
        self.xy_mag_history.append(mag)

    def update_plots(self):
        """Refresh the plots live"""
        if len(self.time_history) > 1:
            self.ax_x.cla()
            self.ax_y.cla()
            self.ax_xy.cla()

            self.ax_x.plot(self.time_history, self.x_history)
            self.ax_y.plot(self.time_history, self.y_history)
            self.ax_xy.plot(self.time_history, self.xy_mag_history)

            self.ax_x.set_title("X Position")
            self.ax_y.set_title("Y Position")
            self.ax_xy.set_title("XY Magnitude")

            for ax in [self.ax_x, self.ax_y, self.ax_xy]:
                ax.grid(True)

        self.canvas.draw()
        self.root.after(100, self.update_plots)

    def clear_plots(self):
        """Clear history + plots"""
        self.time_history.clear()
        self.x_history.clear()
        self.y_history.clear()
        self.xy_mag_history.clear()

    def save_and_clear(self):
        """Save data to CSV for MATLAB/Excel, then clear."""
        if len(self.time_history) == 0:
            print("[TRACK] No data to save.")
            return

        filename = f"ball_tracking_{int(time.time())}.csv"
        data = np.vstack([self.time_history,
                          self.x_history,
                          self.y_history,
                          self.xy_mag_history]).T

        np.savetxt(filename,
                   data,
                   delimiter=",",
                   header="time,x,y,xy_magnitude",
                   comments="")

        print(f"[TRACK] Saved: {filename}")
        self.clear_plots()

def run_plot_gui():
    root = tk.Tk()
    global plot_gui
    plot_gui = TrackingPlotGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

plot_gui = threading.Thread(target = run_plot_gui, daemon = True)
plot_gui.start()

# Start GUI in a separate thread
gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

# Small delay to allow GUI to initialize
time.sleep(0.5)

deadzone_count = 0 

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

    # Display the frame with overlays
    cv2.imshow("Ball Tracking - Real-time Detection", overlay)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Ball tracking stopped.")
        break


    #Calculate the ball position
    found, x, y, radius =  detect.detect_ball(frame)#Placeholder, Kean's function should output a 2D numpy array

    center = np.array(center_point_px)
    # Convert to numpy arrays - only if ball is detected
    if found and x is not None and y is not None:
        # detect_ball is called on the resized (320x240) frame, so x,y are already in the
        # resized coordinate system — do NOT divide by 2 here.
        ball_position = np.array([x, y])
        # Center point (resized and cast to int earlier)
        if plot_gui is not None:
            px = float(x - center[0])
            py = float(y - center[1])
            plot_gui.add_data(px, py)
    else:
        ball_position = None

    print("center point px:", center_point_px[0], center_point_px[1])
    print(f"Ball Position: {ball_position}, Center: {center}")

    # Get current deadzone value
    with pid_gains_lock:
        deadzone_value = current_deadzone

    #Obtain and project the error onto each axis. All inputs are 2D numpy arrays
    error_array, deadzone_count = projected_errors(u1, u2, u3, ball_position, center, deadzone_value, deadzone_count) #output array is: [axis 1, axis 2, axis 3]

    #run each PID controller
    motor1_command = motor1_pid.update(error_array[0])
    motor2_command = motor2_pid.update(error_array[1])
    motor3_command = motor3_pid.update(error_array[2])

    #Send code to each motor by converting the commands to a 3 byte array and sending over serial
    serial_port.send_servo_angles(motor1_command, motor2_command, motor3_command)

    # delay(10)

    
cap.release()
cv2.destroyAllWindows()
