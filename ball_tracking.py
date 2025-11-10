import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread
import queue

class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""
    
    def __init__(self, config_file="config.json"):
        """Initialize ball detector with HSV bounds from config file.
        
        Args:
            config_file (str): Path to JSON config file with HSV bounds and calibration
        """

        # Load configuration from file if it exists
        if os.path.exists("config.json"):
            try:
                with open("config.json", 'r') as f:
                    config = json.load(f)
                
                # Extract HSV color bounds from config
                if 'ball_detection' in config:
                    if config['ball_detection']['lower_hsv']:
                        self.lower_hsv = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
                    if config['ball_detection']['upper_hsv']:
                        self.upper_hsv = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)
                
                # Extract scale factor for position conversion from pixels to meters
                if 'calibration' in config and 'pixel_to_meter_ratio' in config['calibration']:
                    if config['calibration']['pixel_to_meter_ratio']:
                        frame_width = config.get('camera', {}).get('frame_width', 640)
                        self.scale_factor = config['calibration']['pixel_to_meter_ratio'] * (frame_width / 2)
                
                print(f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}")
                print(f"[BALL_DETECT] Scale factor: {self.scale_factor:.6f} m/normalized_unit")
                
            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def detect_ball(self, frame):
        """Detect ball in frame and return detection results.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates of ball center
            radius (float): Ball radius in pixels
            position_m (float): Ball position in meters from center
        """
        # Convert frame from BGR to HSV color space for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create binary mask using HSV color bounds
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Clean up mask using morphological operations
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps
        
        # Find all contours in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, None, 0.0
        
        # Select the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Filter out detections that are too small or too large
        if radius < 5 or radius > 100:
            return False, None, None, 0.0
        
        return True, int(x), int(y), radius

    def draw_detection(self, frame, center_point_px=None, platform_points = [], u_vectors=[], show_info=True):
        """Detect ball and draw detection overlay on frame.
        
        Args:
            frame: Input BGR image frame
            show_info (bool): Whether to display position information text
            
        Returns:
            frame_with_overlay: Frame with detection drawn
            found: True if ball detected
        """
        # Perform ball detection
        found, cx, cy, radius = self.detect_ball(frame)
        
        # Create overlay copy for drawing
        overlay = frame.copy()
        
        if center_point_px is not None:
            cv2.circle(overlay, (center_point_px[0], center_point_px[1]), 1, (0, 0, 255), 6)
            cv2.putText(overlay, f"Center: ({center_point_px[0]}, {center_point_px[1]})", (center_point_px[0] + 10, center_point_px[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #  plot center point
        cv2.circle(overlay, (cx,cy), 1, (255, 255, 255), 1)

        # Draw geometry points and triangle
        for pt in platform_points:
            cv2.circle(overlay, pt, 6, (0, 255, 0), -1)
        if len(platform_points) == 3:
            pts = np.array(platform_points, np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Draw unit vectors from each corner point toward center
            vector_length = 50  # pixels
            u_vectors
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
            
            for i, (pt, unit_vec, color) in enumerate(zip(platform_points, u_vectors, colors)):
                if len(unit_vec) > 0:
                    # Calculate end point of unit vector
                    end_x = int(pt[0] + unit_vec[0] * vector_length)
                    end_y = int(pt[1] + unit_vec[1] * vector_length)
                    end_pt = (end_x, end_y)
                    
                    # Draw arrow from point to end point
                    cv2.arrowedLine(overlay, pt, end_pt, color, 2, tipLength=0.3)
                    
                    # Label the vector
                    cv2.putText(overlay, f"u{i+1}", (end_x+5, end_y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
 
        if found:
            # Draw circle around detected ball
            cv2.circle(overlay, (cx,cy), int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(overlay, (cx,cy), 3, (0, 255, 0), -1)  # Green center dot
            
            if show_info:
                # Display ball position information
                cv2.putText(overlay, f"x: {cx}, y: {cy}", (cx - 30, cy- 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return overlay, found




