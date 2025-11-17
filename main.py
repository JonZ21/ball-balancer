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
import queue

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
Kp = 0.1606
Kd = 0.1376
Ki = 0.0688

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

#Shared data for multithreading
frame_queue = queue.Queue(maxsize=1)
vision_data_lock = threading.Lock()
vision_data = {
    "overlay": None,
    "ball_position": None,
    "found": False,
}
stop_event = threading.Event()
center = np.array(center_point_px)
control_interval = 0.01  # 100 Hz control loop

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

# Multithreaded workers
def capture_frames():
    """Continuously capture frames from the camera."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))

        try:
            frame_queue.put(frame, timeout=0.01)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put(frame, timeout=0.01)
            except queue.Full:
                pass


def vision_processing():
    """Process frames to detect the ball and prepare overlays."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        overlay, found_overlay = detect.draw_detection(
            frame, center_point_px, platform_points, u_vectors, show_info=True
        )

        if found_overlay:
            cv2.putText(
                overlay,
                "Ball Detected: YES",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                overlay,
                "Ball Detected: NO",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        with pid_gains_lock:
            display_kp = current_kp
            display_ki = current_ki
            display_kd = current_kd
            display_deadzone = current_deadzone

        cv2.putText(
            overlay,
            f"Kp: {display_kp:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Ki: {display_ki:.4f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Kd: {display_kd:.4f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Deadzone: {display_deadzone:.1f} px",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.circle(overlay, center_point_px, int(display_deadzone), (255, 0, 255), 2)

        found_ball, x, y, radius = detect.detect_ball(frame)
        if found_ball and x is not None and y is not None:
            ball_position = np.array([x, y])
        else:
            ball_position = None

        with vision_data_lock:
            vision_data["overlay"] = overlay
            vision_data["ball_position"] = ball_position
            vision_data["found"] = found_ball


def control_loop():
    """Compute PID commands and send them to the servos."""
    last_time = time.time()
    while not stop_event.is_set():
        with vision_data_lock:
            ball_position = vision_data["ball_position"]

        with pid_gains_lock:
            deadzone_value = current_deadzone

        error_array = projected_errors(u1, u2, u3, ball_position, center, deadzone_value)

        now = time.time()
        dt = max(now - last_time, 1e-3)
        last_time = now

        motor1_command = motor1_pid.update(error_array[0], dt=dt)
        motor2_command = motor2_pid.update(error_array[1], dt=dt)
        motor3_command = motor3_pid.update(error_array[2], dt=dt)

        print("center point px:", center_point_px[0], center_point_px[1])
        print(f"Ball Position: {ball_position}, Center: {center}")

        serial_port.send_servo_angles(motor1_command, motor2_command, motor3_command)
        time.sleep(control_interval)

# Start GUI in a separate thread
gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

# Small delay to allow GUI to initialize
time.sleep(0.5)

# Start worker threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
vision_thread = threading.Thread(target=vision_processing, daemon=True)
control_thread = threading.Thread(target=control_loop, daemon=True)

capture_thread.start()
vision_thread.start()
control_thread.start()

try:
    while not stop_event.is_set():
        with vision_data_lock:
            overlay = vision_data["overlay"]

        if overlay is not None:
            cv2.imshow("Ball Tracking - Real-time Detection", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Ball tracking stopped.")
            stop_event.set()
            break
finally:
    stop_event.set()
    capture_thread.join(timeout=1)
    vision_thread.join(timeout=1)
    control_thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
