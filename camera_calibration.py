#obtain calibration data necessary to test ball tracking on the stewart platform. 
#most functions are from the 1D balancer simple_cal file, but with the motor calibration removed. 

import cv2
import numpy as np
import json
import math
from datetime import datetime

class CameraCalibrator:
    """Interactive calibration tool for ball tracking on a 3D platform."""

    def __init__(self):
        # Camera setup
        self.CAM_INDEX = 1
        self.FRAME_W, self.FRAME_H = 640, 480

        # Calibration state
        self.current_frame = None
        self.phase = "color"  # "color", "geometry", "complete"

        # Color calibration data
        self.hsv_samples = []
        self.lower_hsv = None
        self.upper_hsv = None

        # Geometry calibration data
        self.platform_points = []  # 3 clicked points
        self.center_point = None
        self.pixel_to_meter_ratio = None

        # Unit Vectors (return as np arrays for easier calculations)
        self.u1 = []
        self.u2 = []
        self.u3 = []

        # Known distance between first two points in meters
        self.KNOWN_RADIUS_M = 0.15  # adjust for your platform

    # ---------------- Mouse Interaction ----------------
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks during calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                self.sample_color(x, y)

            elif self.phase == "geometry" and len(self.platform_points) < 3:
                self.platform_points.append((x, y))
                print(f"[GEO] Point {len(self.platform_points)} selected at ({x}, {y})")

                if len(self.platform_points) == 3:
                    self.calculate_geometry()
                    self.phase = "complete"
                    print(f"[INFO] Geometry done. Center computed automatically: {self.center_point}")
                    print("[INFO] Press 's' to save config.")

    # ---------------- Color Sampling ----------------
    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point."""
        if self.current_frame is None:
            return

        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        region = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    region.append(hsv[py, px])

        if region:
            self.hsv_samples.extend(region)
            samples = np.array(self.hsv_samples)

            # Add margins for robustness
            h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)

            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]
            print(f"[COLOR] Sampled {len(self.hsv_samples)} pixels.")

    # ---------------- Geometry Calculation ----------------
    def calculate_geometry(self):
        """Compute pixel-to-meter ratio and centroid from 3 clicked points."""
        p1, p2, p3 = self.platform_points

        print("p1, p2, p3:", p1, p2, p3)
        # Compute centroid (platform center)
        cx = (p1[0] + p2[0] + p3[0]) / 3
        cy = (p1[1] + p2[1] + p3[1]) / 3
        tri_radius = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / 2
        self.center_point = (int(cx), int(cy))

        # Compute unit vectors
        
        mag_1 = math.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        mag_2 = math.sqrt((cx - p2[0])**2 + (cy - p2[1])**2)
        mag_3 = math.sqrt((cx - p3[0])**2 + (cy - p3[1])**2)

        self.u1 = np.array([(cx - p1[0])/mag_1, (cy - p1[1])/mag_1])
        self.u2 = np.array([(cx - p2[0])/mag_2, (cy - p2[1])/mag_2])
        self.u3 = np.array([(cx - p3[0])/mag_3, (cy - p3[1])/mag_3])

        # Compute pixel-to-meter ratio using first two points
        pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.pixel_to_meter_ratio = self.KNOWN_RADIUS_M / pixel_dist

        print(f"[GEO] Pixel distance = {pixel_dist:.2f}")
        print(f"[GEO] Ratio = {self.pixel_to_meter_ratio:.6f} m/pixel")
        print(f"[GEO] Computed platform center = ({self.center_point[0]}, {self.center_point[1]})")

    # ---------------- Save Config ----------------
    def save_config(self):
        """Save calibration results to config.json."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H),
                "center_point_px": [int(self.center_point[0]), int(self.center_point[1])] if self.center_point else None
            },
            "ball_detection": {
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv else None
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "unit_vectors": {
                    "u1": [float(x) for x in self.u1] if len(self.u1) > 0 else None,
                    "u2": [float(x) for x in self.u2] if len(self.u2) > 0 else None,
                    "u3": [float(x) for x in self.u3] if len(self.u3) > 0 else None
                }
            }
        }

        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    # ---------------- Visualization ----------------
    def draw_overlay(self, frame):
        """Draw calibration info and geometry overlay."""
        overlay = frame.copy()
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Instruction text
        if self.phase == "color":
            msg = "Click on ball to sample color. Press 'c' when done."
        elif self.phase == "geometry":
            msg = "Click three platform corners."
        elif self.phase == "complete":
            msg = "Press 's' to save configuration."
        else:
            msg = ""
        cv2.putText(overlay, msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show sampled HSV count
        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw geometry points and triangle
        for pt in self.platform_points:
            cv2.circle(overlay, pt, 6, (0, 255, 0), -1)
        if len(self.platform_points) == 3:
            pts = np.array(self.platform_points, np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Draw unit vectors from each corner point toward center
            vector_length = 50  # pixels
            unit_vectors = [self.u1, self.u2, self.u3]
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: Blue, Green, Red
            
            for i, (pt, unit_vec, color) in enumerate(zip(self.platform_points, unit_vectors, colors)):
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
        
        if self.center_point:
            cv2.circle(overlay, self.center_point, 8, (0, 255, 255), -1)
            cv2.putText(overlay, "Center", (self.center_point[0]+10, self.center_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return overlay

    # ---------------- Main Loop ----------------
    def run(self):
        """Main interactive calibration loop."""
        cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        print("[INFO] Calibration started")
        print("1. Click on ball to sample color, press 'c' when done.")
        print("2. Click 3 corners on the platform.")
        print("3. Press 's' to save, or 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            self.current_frame = frame
            overlay = self.draw_overlay(frame)
            cv2.imshow("Calibration", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.phase == "color" and self.hsv_samples:
                self.phase = "geometry"
                print("[INFO] Color calibration done. Click 3 platform corners.")
            elif key == ord('s') and self.phase == "complete":
                self.save_config()
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrator = CameraCalibrator()
    calibrator.run()

