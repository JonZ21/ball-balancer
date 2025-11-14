# nm_tuner.py — super simple Nelder–Mead tuner for (Kp, Ki, Kd)
#
# What it does (high level):
# - picks an initial "simplex" of 4 gain triples around your current gains
# - tries one 10 s trial per vertex, scoring with your J = w1*IAE + w2*P95
# - then repeats classic NM steps: reflect → (maybe) expand → (maybe) contract → else shrink
# - stops when J values & vertex motions get small, or when max_iter reached
#
# Why Nelder–Mead? It’s a derivative-free “direct search” — perfect when the only way
# to judge gains is “run the real rig for 10 s and get a score”. :contentReference[oaicite:1]{index=1}

import time, threading
import numpy as np
from typing import Tuple
from tuning_j import start_trial, finish_and_score

# standard NM coefficients (don’t overthink these) :contentReference[oaicite:2]{index=2}
_ALPHA = 1.0   # reflection
_GAMMA = 2.0   # expansion
_RHO   = 0.5   # contraction
_SIGMA = 0.5   # shrink

def _apply_gains(motor_pids, kp: float, ki: float, kd: float) -> None:
    """Set the same (Kp, Ki, Kd) on all 3 motors and clear integrators."""
    for pid in motor_pids:
        pid.update_gains(Kp=kp, Ki=ki, Kd=kd)
        pid.reset_integral()

def _run_trial_get_J(motor_pids, trial_sec: float, gains: np.ndarray,
                     w1: float, w2: float, pctl: int) -> float:
    """Set gains → run a timed trial → return scalar J (lower is better)."""
    kp, ki, kd = map(float, gains)
    # keep search sane with light clamping (adjust to your safe ranges)
    kp = float(np.clip(kp, 0.0, 1.5))
    ki = float(np.clip(ki, 0.0, 0.8))
    kd = float(np.clip(kd, 0.0, 1.5))

    _apply_gains(motor_pids, kp, ki, kd)

    # scorer buffers are filled by your main loop calling log_sample(xy_error) each frame
    start_trial()
    time.sleep(trial_sec)         # non-blocking w.r.t. your main loop
    J, parts = finish_and_score(w1=w1, w2=w2, pctl=pctl)
    if J is None:
        return 1e12  # penalize failures (e.g., not enough samples)

    print(f"[NM] try  Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}  ->  "
          f"J={J:.3f}  IAE={parts['IAE']:.2f}  P{pctl}={parts[f'P{pctl}']:.2f}")
    return float(J)

def _make_initial_simplex(x0: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Construct 4 vertices (for 3 params) around x0 by nudging one param per vertex."""
    S = np.tile(x0, (x0.size + 1, 1))   # shape (4,3)
    for i in range(x0.size):
        S[i+1, i] += step[i]
    return S

def _nm_minimize(f, x0: np.ndarray, step: np.ndarray,
                 max_iter: int = 20, tol_f: float = 1e-3, tol_x: float = 1e-3) -> Tuple[np.ndarray, float]:
    """Minimal, readable Nelder–Mead (no bells/whistles)."""
    S = _make_initial_simplex(x0, step)            # vertices
    F = np.array([f(v) for v in S], dtype=float)   # scores at vertices

    for it in range(max_iter):
        # 1) order vertices (best first)
        order = np.argsort(F)
        S, F = S[order], F[order]

        print(f"[NM] iter {it+1:02d}  bestJ={F[0]:.3f}")
        
        # stop if scores are nearly equal AND the simplex is tiny
        if np.std(F) < tol_f and np.max(np.linalg.norm(S - S[0], axis=1)) < tol_x:
            break

        # centroid of best 3 (exclude worst)
        centroid = np.mean(S[:-1], axis=0)
        worst    = S[-1]

        # 2) reflect worst across centroid
        x_r = centroid + _ALPHA * (centroid - worst)
        f_r = f(x_r)

        if F[0] <= f_r < F[-2]:
            S[-1], F[-1] = x_r, f_r
            continue

        if f_r < F[0]:
            # 3) expand
            x_e = centroid + _GAMMA * (x_r - centroid)
            f_e = f(x_e)
            S[-1], F[-1] = (x_e, f_e) if f_e < f_r else (x_r, f_r)
            continue

        # 4) contract (pull worst towards centroid)
        x_c = centroid + _RHO * (worst - centroid)
        f_c = f(x_c)
        if f_c < F[-1]:
            S[-1], F[-1] = x_c, f_c
            continue

        # 5) shrink towards best (last resort if all else failed)
        best = S[0]
        for i in range(1, S.shape[0]):
            S[i] = best + _SIGMA * (S[i] - best)
            F[i] = f(S[i])

    # return best vertex
    order = np.argsort(F)
    return S[order][0], float(F[order][0])

def start_nm_tuning(
    motor_pids,                   # tuple/list: (motor1_pid, motor2_pid, motor3_pid)
    trial_sec: float = 10.0,      # one trial length (seconds)
    w1: float = 1.0, w2: float = 0.7, pctl: int = 95,   # J = w1*IAE + w2*P95
    x0: Tuple[float, float, float] = (0.16, 0.07, 0.14),# initial (Kp,Ki,Kd)
    step: Tuple[float, float, float] = (0.05, 0.03, 0.05),
    max_iter: int = 20
) -> None:
    """Launch tuner in a thread so your GUI/video/serial keep running."""
    def worker():
        print("[NM] starting Nelder–Mead tuner …")
        x0_np, step_np = np.array(x0, float), np.array(step, float)

        def f_eval(g):  # g = [Kp, Ki, Kd]
            return _run_trial_get_J(motor_pids, trial_sec, np.array(g, float), w1, w2, pctl)

        best_g, best_J = _nm_minimize(f_eval, x0_np, step_np, max_iter=max_iter)
        kp, ki, kd = map(float, best_g)
        print(f"[NM] BEST  Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f}   ->   J={best_J:.3f}")
        _apply_gains(motor_pids, kp, ki, kd)  # leave the robot with best gains

    threading.Thread(target=worker, daemon=True).start()
