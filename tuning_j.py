"""
Performance Scoring Module for Ball Balancer PID Tuning

This module provides tools to quantitatively evaluate PID controller performance
during controlled trials. It computes a composite score based on:
  - IAE (Integral of Absolute Error): Measures accumulated error over time
  - Percentile Error: Measures worst-case performance (default P95)

TYPICAL WORKFLOW:
  1. Press 's' key → calls start_trial() → begins data collection
  2. Main loop runs for ~10 seconds, calling log_sample(xy_error) each frame
  3. Press 'e' key → calls finish_and_score() → computes J score
  4. Lower J score = better performance

SCORE FORMULA:
  J = w1 * IAE + w2 * P95
  where:
    - IAE = sum of all radial errors * dt (integrated error)
    - P95 = 95th percentile of radial errors (measures peak errors)
    - w1, w2 = weights to balance steady-state vs transient performance

USE CASES:
  - Manual tuning: Compare different gain settings
  - Auto-tuning: Nelder-Mead optimizer uses J to find optimal gains
"""

import time
import numpy as np

# ============================================================================
# GLOBAL STATE VARIABLES
# ============================================================================
# These track the current trial's data. Only one trial can be active at a time.

T: list[float] = []   # Timestamps: time elapsed since trial start [seconds]
E: list[float]  = []  # Errors: radial distance from center [pixels]
trial_active: bool = False  # Flag: is a trial currently running?
t0: float = 0.0  # Trial start time (absolute timestamp from time.time())

def start_trial():
    """Begin a new performance evaluation trial.
    
    This function:
      1. Clears any previous trial data
      2. Records the current time as trial start
      3. Activates data logging
    
    After calling this, log_sample() will begin collecting error data
    until finish_and_score() is called.
    
    NOTE: If a trial is already active, this will restart it from scratch.
    """
    global T, E, trial_active, t0
    
    # Warn if restarting an active trial
    if trial_active:
        print("[TUNE] Warning: Restarting trial (previous trial data will be lost)")
    
    # Clear all previous data
    T.clear()
    E.clear()
    
    # Mark trial as active and record start time
    trial_active = True
    t0 = time.time()
    
    print(f"[TUNE] Trial started at t={t0:.2f}")

def log_sample(xy_error):
    """Log one error measurement from the current video frame.
    
    This should be called once per frame in your main control loop.
    It calculates the radial error (distance from target) and logs it
    with a timestamp.
    
    Args:
        xy_error: 2D numpy array [x_error, y_error] in pixels
                  Represents the ball's displacement from center point
                  If ball detection fails, pass the last valid error
    
    Behavior:
        - If no trial is active: Does nothing (silent no-op)
        - If trial is active: Appends timestamp and radial error to logs
    
    Example:
        >>> ball_pos = np.array([160, 120])
        >>> center = np.array([150, 115])
        >>> xy_error = ball_pos - center  # [10, 5]
        >>> log_sample(xy_error)  # Logs radial error = sqrt(10² + 5²) ≈ 11.18 px
    """
    # Do nothing if no trial is running
    if not trial_active:
        return
    
    # Validate input
    if xy_error is None or not isinstance(xy_error, np.ndarray):
        print("[TUNE] Warning: Invalid xy_error passed to log_sample (skipping sample)")
        return
    
    # Record elapsed time since trial start
    elapsed_time = time.time() - t0
    T.append(elapsed_time)
    
    # Calculate radial error: distance from center = sqrt(x² + y²)
    radial_error = float(np.linalg.norm(xy_error))
    E.append(radial_error)

def finish_and_score(w1=1.4, w2=0.3, w3=1.2, pctl=95):
    """Complete the trial and compute the performance score.
    
    This function:
      1. Stops data collection
      2. Validates collected data
      3. Computes performance metrics
      4. Returns composite score J
    
    Args:
        w1: Weight for IAE (Integral Absolute Error)
            Higher w1 = penalize steady-state error more
            Typical range: 0.5 - 2.0
        
        w2: Weight for percentile error
            Higher w2 = penalize transient spikes more
            Typical range: 0.3 - 1.5
        
        pctl: Percentile to use (e.g., 95 = P95)
              95 = ignore worst 5% of errors
              90 = ignore worst 10% of errors
              Typical range: 90 - 99
    
    Returns:
        Tuple of (J, parts) where:
          J: Composite score (float) - LOWER IS BETTER
             None if insufficient data
          
          parts: Dictionary with breakdown:
            - 'IAE': Integral of absolute error
            - 'P95': 95th percentile error (or other pctl)
            - 'dt': Time step used
            - 'N': Number of samples collected
    
    Score Interpretation:
        J < 50:  Excellent performance
        J < 100: Good performance
        J < 200: Acceptable performance
        J > 200: Poor performance (needs tuning)
    """
    global trial_active
    
    # Mark trial as inactive
    trial_active = False
    
    # Validate we have enough data for meaningful statistics
    MIN_SAMPLES = 5
    if len(T) < MIN_SAMPLES:
        print(f"[TUNE] Error: Insufficient samples collected")
        print(f"       Got {len(T)} samples, need at least {MIN_SAMPLES}")
        print(f"       Trial may have been too short or detection failed")
        return None, None
    
    # Convert lists to numpy arrays for efficient computation
    timestamps = np.asarray(T, dtype=float)
    radial_errors = np.asarray(E, dtype=float)
    
    # Estimate time step between samples
    # Assumes ~30 FPS camera. Could also use: dt = np.mean(np.diff(timestamps))
    dt = 1.0 / 30.0  # seconds per frame
    
    # ---- Compute IAE (Integral of Absolute Error) ----
    # This measures total accumulated error over the trial
    # Formula: IAE = Σ |error_i| * Δt
    # Units: pixel-seconds
    IAE = np.sum(radial_errors) * dt
    
    # ---- Compute Percentile Error ----
    # This measures peak/worst-case performance
    # P95 = value below which 95% of errors fall
    # Units: pixels
    Pctl = np.percentile(radial_errors, pctl)
    if len(radial_errors) > 1:
        NEAR_CENTER_THRESHOLD = 22.0  # pixels - only count oscillations near center
        error_changes = np.abs(np.diff(radial_errors))  # |e[i+1] - e[i]|

        # Only count oscillations when BOTH adjacent errors are near center
        # This prevents penalizing large corrections when ball is far from center
        near_center_mask = (radial_errors[:-1] < NEAR_CENTER_THRESHOLD) & (radial_errors[1:] < NEAR_CENTER_THRESHOLD)
        OSC = np.sum(error_changes[near_center_mask])  # Total variation near center only
    else:
        OSC = 0.0

# ---- Compute Composite Score J ----
# Weighted combination of steady-state, transient, and oscillation performance
    J = w1 * IAE + w2 * Pctl + w3 * OSC

# Package detailed results
    parts = {
        "IAE": float(IAE),
        f"P{pctl}": float(Pctl),
        "OSC": float(OSC),
        "dt": float(dt),
        "N": int(len(radial_errors)),
        "duration": float(timestamps[-1]),  # Total trial duration
        "mean_error": float(np.mean(radial_errors)),
        "max_error": float(np.max(radial_errors)),
        "min_error": float(np.min(radial_errors))
    }
    
    return J, parts

