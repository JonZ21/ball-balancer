"""_summary_

This code should compute a scalar score for 
performance of the balancer based on a 10s trial.


What it does:
- starts a trial
- each frame: log time vs xy-error
- Compute J = w1*IAE + w2*PEAK

How to use:
- call start_trial() to begin
- per frame: log_sample(numpy array of xy_error)
- when done: ouputs J. 
"""

from typing import Optional, Tuple, Dict
import time
import numpy as np
import csv

# Lists to log data
T: list[float] = []   # time since start [s]
E: list[float]  = []  # radial error r_k = ||xy_error|| 
trial_active = False
t0 = 0.0 # Start time of trial

def start_trial() -> None:
    """Start a new trial for performance evaluation."""
    global T, E, trial_active, t0

    #Reset logs
    T.clear(); E.clear();
    trial_active = True
    t0 = time.time()

def log_sample(xy_error: np.ndarray) -> None:
    """ 
    Add sample for current frame at this time.

    2D error. if ball not found, pass last valid error.
    """

    if not trial_active:
        return
    
    T.append(time.time() - t0)
    E.append(float(np.linalg.norm(xy_error)))

def finish_and_score(w1=1.0, w2=0.7, pctl=95):
    """
    Camera-only score on raw r[k]:
      J = w1*IAE + w2*Pctl
    where:
      IAE  = sum_k r[k]*dt
      Pctl = percentile(r, pctl)  # e.g., P95
    """
    global trial_active
    trial_active = False
    if len(T) < 5:
        return None, None
    
    t = np.asarray(T, dtype=float)
    r = np.asarray(E, dtype=float) #radial error

    dt = 1/30

    IAE = np.sum(r)*dt
    Pctl = np.percentile(r, pctl)
    J = w1*IAE + w2*Pctl

    parts = {"IAE": IAE, f"P{pctl}": Pctl, "dt": dt, "N": int(len(r))}
    return J, parts

