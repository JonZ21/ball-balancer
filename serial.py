import serial
import time
import numpy as np

class SerialPort:
    def __init__(self):
        # Servo hardware configuration
        self.serial = None  # Serial connection to servo
        # Note: On Windows use "COM3", on macOS/Linux use "/dev/tty.usbserial-*" or "/dev/ttyUSB*"
        # You can find the correct port in Arduino IDE or by listing ports: python -m serial.tools.list_ports
        self.servo_port = "COM3"  # Servo communication port - CHANGE THIS TO YOUR ACTUAL PORT

    def connect_serial(self):
        #Establish serial connection to servo motor for automated limit finding.
            
        #Returns:
            #bool: True if connection successful, False otherwise
        
        try:
            self.serial = serial.Serial(self.servo_port, 9600)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO] Connected")
            return True
        except:
            print("[SERVO] Failed to connect")
            return False

    def send_servo_angles(self, angle1, angle2, angle3):
        #Send angle command to servo motor with safety clipping.
        
        #Args:
            #angle1, angle2, angle3 (float): Desired servo angles in degrees (0-30)

        if self.serial and self.serial.is_open: #Only run if the serial port has been connected
            # Clip angles to safe range and convert to integers
            angles = [
                int(np.clip(angle1, min_angle, max_angle)), 
                int(np.clip(angle2, min_angle, max_angle)), 
                int(np.clip(angle3, min_angle, max_angle))
            ]
            # Send as 3 bytes (one byte per angle, values 0-30)
            self.serial.write(bytes(angles))
            self.serial.flush()  # Ensure data is sent immediately
    
    def read_serial_output(self):
        #Read and return any available data from the serial port.
        
        #Returns:
            #str: Serial output string, or empty string if no data available
        
        if self.serial and self.serial.is_open:
            if self.serial.in_waiting > 0:
                try:
                    # Read all available bytes and decode to string
                    data = self.serial.readline().decode('utf-8').strip()
                    return data
                except:
                    return ""
        return ""