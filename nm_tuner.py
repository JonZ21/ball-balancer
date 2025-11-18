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

import time
import threading
import numpy as np
from tuning_j import start_trial, finish_and_score

# standard NM coefficients (don't overthink these)
ALPHA = 1.0   # reflection
GAMMA = 2.0   # expansion
RHO   = 0.5   # contraction
SIGMA = 0.5   # shrink

KP_MIN, KP_MAX = 0.0, 2
KI_MIN, KI_MAX = 0.0, 0.8
KD_MIN, KD_MAX = 0.0, 2

def clip_gains(kp, ki, kd):
    """Clamp PID gains to safe, physically reasonable bounds.
    
    WHY THIS IS NECESSARY:
      - Protects hardware from extreme control outputs
      - Prevents optimizer from exploring unrealistic parameter space
      - Keeps integral windup manageable
      - Ensures numerical stability
    
    BOUNDS RATIONALE:
      Kp ∈ [0, 1.5]: Proportional gain
        - Too low: Sluggish response
        - Too high: Oscillation, overshoot
      
      Ki ∈ [0, 0.8]: Integral gain
        - Too low: Steady-state error
        - Too high: Windup, instability
      
      Kd ∈ [0, 1.5]: Derivative gain
        - Too low: No damping
        - Too high: Noise amplification
    
    Args:
        kp, ki, kd: Proposed gain values (may be out of bounds)
    
    Returns:
        Tuple of (kp, ki, kd) clipped to safe ranges
    
    Example:
        >>> clip_gains(2.5, -0.1, 0.5)  # Invalid gains
        (1.5, 0.0, 0.5)  # Clipped to valid range
    """
    # Clip each gain independently to its valid range
    kp_clipped = float(np.clip(kp, KP_MIN, KP_MAX))
    ki_clipped = float(np.clip(ki, KI_MIN, KI_MAX))
    kd_clipped = float(np.clip(kd, KD_MIN, KD_MAX))
    
    # Warn if clipping occurred (helps debug optimizer issues)
    if kp != kp_clipped or ki != ki_clipped or kd != kd_clipped:
        print(f"[NM] Clipped gains: ({kp:.4f}, {ki:.4f}, {kd:.4f}) → "
              f"({kp_clipped:.4f}, {ki_clipped:.4f}, {kd_clipped:.4f})")
    
    return kp_clipped, ki_clipped, kd_clipped

def apply_gains(motor_pids, kp, ki, kd, reset_integral=True, 
                gains_lock=None, update_globals_fn=None):
    """Set the same (Kp, Ki, Kd) on all 3 motors and optionally clear integrators.
    
    Args:
        motor_pids: List of PID controllers
        kp, ki, kd: Gain values
        reset_integral: Whether to reset integral term
        gains_lock: Threading lock for thread safety (optional)
        update_globals_fn: Callback to update global variables (optional)
    """
    if gains_lock is not None and update_globals_fn is not None:
        with gains_lock:
            update_globals_fn(kp, ki, kd)
    
    for pid in motor_pids:
        pid.update_gains(Kp=kp, Ki=ki, Kd=kd)
        if reset_integral:
            pid.reset_integral()

def run_trial(pid_list, trial_duration_s, w1, w2, w3, pctl, gains):
    """Execute one complete performance trial with specified PID gains.
    
    This is the core evaluation function used by the Nelder-Mead optimizer.
    Each call performs a full trial on the actual hardware:
    
    TRIAL SEQUENCE:
      1. Validate and clip proposed gains to safe bounds
      2. Apply gains to all three motor PIDs
      3. Reset PID integral terms (ensures fair comparison)
      4. Signal trial start to scoring module
      5. Wait for trial_duration_s while system runs
      6. Collect performance score J
      7. Return score to optimizer
    
    IMPORTANT NOTES:
      - This function sleeps but doesn't block the main control loop
      - The main loop must call log_sample() each frame during sleep
      - Each trial is independent (integrators reset)
      - Failed trials return large penalty score
    
    Args:
        pid_list: List of 3 PIDcontroller objects [motor1, motor2, motor3]
        trial_duration_s: How long to run trial (typically 10 seconds)
        w1: Weight for IAE in score calculation
        w2: Weight for percentile error in score calculation
        pctl: Percentile to use (e.g., 95 for P95)
        gains: Tuple of (Kp, Ki, Kd) to test
    
    Returns:
        J: Performance score (float)
           - Lower is better
           - 1e12 if trial failed (penalty)
           - Typical good scores: 30-100
           - Typical bad scores: 200+
    
    Example:
        >>> score = run_trial([pid1, pid2, pid3], 10.0, 1.0, 0.7, 95, (0.2, 0.1, 0.15))
        [NM] try  Kp=0.2000 Ki=0.1000 Kd=0.1500  ->  J=67.342
        >>> print(f"Trial score: {score}")
        Trial score: 67.342
    """
    # Step 1: Ensure gains are within safe bounds
    kp, ki, kd = clip_gains(*gains)
    
    # Step 2: Apply gains to all PIDs and reset integral windup
    # (Reset ensures each trial starts from same state)
    apply_gains(pid_list, kp, ki, kd, reset_integral=True)
    
    print(f"[NM] Starting trial with Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}...")
    
    # Give the operator a short countdown so they can prepare the rig.
    # Default countdown seconds can be adjusted here.
    COUNTDOWN_SECONDS = 5
    try:
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"[NM] Trial starts in {remaining} second(s)...", end='\r', flush=True)
            time.sleep(1)
        print()  # move to next line after countdown
    except KeyboardInterrupt:
        # If the operator hits Ctrl-C, proceed immediately
        print("\n[NM] Countdown interrupted; starting trial immediately.")

    # Step 3: Signal scoring module to begin data collection
    start_trial()

    # Step 4: Wait for trial to complete
    # NOTE: This sleep doesn't block the main OpenCV loop (different thread)
    #       Main loop continues calling log_sample() during this time
    time.sleep(trial_duration_s)
    
    # Step 5: Signal scoring module to stop and compute score
    J, parts = finish_and_score(w1=w1, w2=w2, w3=w3, pctl=pctl)
    
    # Step 6: Handle trial failure
    if J is None:
        print("[NM] ERROR: Trial produced no valid score")
        print("     Possible causes:")
        print("       - Ball detection failed")
        print("       - Camera disconnected")
        print("       - Trial too short")
        print("     Returning penalty score: 1e12")
        return 1e12
    
    # Step 7: Report results
    percentile_key = f"P{pctl}"
    print(f"[NM] TRIAL COMPLETE:")
    print(f"     Gains: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}")
    print(f"     Score: J={J:.3f}")
    print(f"     Breakdown: IAE={parts['IAE']:.2f}, {percentile_key}={parts[percentile_key]:.2f}")
    print(f"     Samples: N={parts['N']}, Duration={parts.get('duration', 'N/A'):.2f}s")
    
    return float(J)

def make_initial_simplex(x0, scale=0.3):
    """
    Create an initial simplex around x0 by nudging each dimension by 'scale'.

    For 3D (Kp, Ki, Kd), returns shape (4,3):
      row 0 = x0
      row 1 = x0 + [scale, 0, 0]
      row 2 = x0 + [0, scale, 0]
      row 3 = x0 + [0, 0, scale]
    """
    n = len(x0)
    S = np.tile(np.asarray(x0, float), (n + 1, 1))
    for i in range(n):
        S[i + 1, i] += scale if x0[i] == 0 else scale * abs(x0[i])
    return S

def nelder_mead_minimize(f, x0, scale=0.3, max_iter=20, tol_f=1e-3, tol_x=1e-3):
    """Nelder-Mead optimization algorithm explained simply.
    
    WHAT IS A SIMPLEX?
      Think of it like a "tetrahedron of guesses" in 3D space:
        - For 3 gains (Kp, Ki, Kd), we need 4 points (vertices)
        - Each point is a complete set of gains to test
        - The simplex moves/morphs to find better gains
    
    HOW IT WORKS (high level):
      1. Start with 4 gain combinations around your initial guess
      2. Test all 4 by running trials → get 4 scores
      3. Find the WORST point
      4. Try to replace it with something better by:
         - Reflecting it away from the other points
         - Maybe expanding further if that works well
         - Or contracting back if reflection fails
      5. Repeat until scores stop improving
    
    WHY THIS IS SMART:
      - No calculus needed (can't take derivatives of "run hardware trial")
      - Adapts to the landscape (expands when good, contracts when stuck)
      - Only needs ~20 trials to find good gains
    
    Variables:
      S: Array of simplex vertices (4 rows × 3 columns)
         Each row = one set of (Kp, Ki, Kd) to try
      F: Array of scores for each vertex (4 numbers)
         F[0] = score for S[0], F[1] = score for S[1], etc.
    """
    
    # ========================================================================
    # STEP 1: Create initial simplex (4 starting points)
    # ========================================================================
    # Start with your initial guess x0, then nudge each gain slightly
    # Example: if x0 = [0.16, 0.07, 0.14]
    #   S[0] = [0.16, 0.07, 0.14]  ← original
    #   S[1] = [0.26, 0.07, 0.14]  ← nudged Kp
    #   S[2] = [0.16, 0.17, 0.14]  ← nudged Ki
    #   S[3] = [0.16, 0.07, 0.24]  ← nudged Kd
    S = make_initial_simplex(x0, scale=scale)
    
    # ========================================================================
    # STEP 2: Evaluate initial simplex (run 4 trials)
    # ========================================================================
    # f(v) calls run_trial() which runs the hardware for 10s and returns score J
    # This takes ~40 seconds (4 trials × 10s each)
    F = np.array([f(vertex) for vertex in S], dtype=float)
    
    print("[NM] Initial simplex (4 starting guesses):")
    for i in range(S.shape[0]):
        print(f"  Vertex {i}: Kp={S[i][0]:.4f} Ki={S[i][1]:.4f} Kd={S[i][2]:.4f}  →  J={F[i]:.3f}")
    
    # ========================================================================
    # STEP 3: Main optimization loop
    # ========================================================================
    for iteration in range(max_iter):
        
        # --------------------------------------------------------------------
        # Sort vertices from best (lowest J) to worst (highest J)
        # --------------------------------------------------------------------
        # np.argsort gives indices that would sort the array
        # Example: if F = [100, 50, 200, 75], then idx = [1, 3, 0, 2]
        #   meaning F[1]=50 is best, F[3]=75 is second, etc.
        indices = np.argsort(F)
        S = S[indices]  # Reorder simplex vertices
        F = F[indices]  # Reorder scores to match
        
        # Now: S[0] = best vertex, S[-1] = worst vertex
        #      F[0] = best score,  F[-1] = worst score
        
        print(f"[NM] Iteration {iteration+1}/{max_iter}:")
        print(f"     Best score so far: J={F[0]:.3f} at Kp={S[0][0]:.4f} Ki={S[0][1]:.4f} Kd={S[0][2]:.4f}")
        
        # --------------------------------------------------------------------
        # Check convergence: Are we done optimizing?
        # --------------------------------------------------------------------
        # Two conditions both must be met:
        
        # Condition 1: All scores are similar (standard deviation is small)
        score_spread = np.std(F)
        scores_converged = score_spread < tol_f
        
        # Condition 2: All vertices are close together (simplex is tiny)
        # Calculate distance from each vertex to the best vertex
        distances_to_best = [np.linalg.norm(S[i] - S[0]) for i in range(len(S))]
        max_distance = np.max(distances_to_best)
        simplex_converged = max_distance < tol_x
        
        if scores_converged and simplex_converged:
            print(f"[NM] ✓ CONVERGED! Score spread={score_spread:.6f}, Max distance={max_distance:.6f}")
            break
        
        # --------------------------------------------------------------------
        # Calculate centroid = average of all GOOD vertices (exclude worst)
        # --------------------------------------------------------------------
        # Think: "center of mass" of the 3 best points
        # Math: centroid = (S[0] + S[1] + S[2]) / 3
        # In code: S[:-1] means "all except last", then take mean
        centroid = np.mean(S[:-1], axis=0)
        
        # Get the worst vertex (highest score)
        worst_vertex = S[-1]
        worst_score = F[-1]
        
        print(f"     Worst vertex: J={worst_score:.3f} at Kp={worst_vertex[0]:.4f} Ki={worst_vertex[1]:.4f} Kd={worst_vertex[2]:.4f}")
        print(f"     Trying to replace it...")
        
        # ====================================================================
        # OPERATION 1: REFLECTION
        # ====================================================================
        # IDEA: "Bounce" the worst point away from the centroid
        # 
        # Visual in 2D:
        #      best1 •
        #           / \
        #          /   \
        #    best2•  C  •worst    →    best1 •
        #                                    / \
        #                                   /   \
        #                             best2• C   •reflected
        #
        # Math: reflected = centroid + ALPHA × (centroid - worst)
        #   where ALPHA=1.0 means "same distance on opposite side"
        #   
        # Example: If centroid=[0.15, 0.08, 0.12] and worst=[0.10, 0.05, 0.08]
        #   reflected = [0.15, 0.08, 0.12] + 1.0 × ([0.15, 0.08, 0.12] - [0.10, 0.05, 0.08])
        #             = [0.15, 0.08, 0.12] + [0.05, 0.03, 0.04]
        #             = [0.20, 0.11, 0.16]
        
        reflected_vertex = centroid + ALPHA * (centroid - worst_vertex)
        reflected_score = f(reflected_vertex)  # Run trial (~10 seconds)
        
        print(f"     → Reflected: J={reflected_score:.3f}")
        
        # Check if reflected point is "medium good"
        # (better than worst, but not better than best)
        best_score = F[0]
        second_worst_score = F[-2]
        
        if best_score <= reflected_score < second_worst_score:
            # Reflection worked! Replace worst with reflected point
            print(f"     ✓ Reflection accepted (medium quality)")
            S[-1] = reflected_vertex
            F[-1] = reflected_score
            continue  # Go to next iteration
        
        # ====================================================================
        # OPERATION 2: EXPANSION
        # ====================================================================
        # IDEA: If reflection was REALLY good, try going even further!
        # 
        # Visual:
        #   best1 •
        #        / \
        #       /   \
        # best2• C   •reflected     →  push further  →  • expanded
        #
        # Math: expanded = centroid + GAMMA × (reflected - centroid)
        #   where GAMMA=2.0 means "go twice as far from centroid"
        #
        # Only try this if reflected point is better than current best
        
        if reflected_score < best_score:
            # Reflection beat our best! Try expanding even further
            expanded_vertex = centroid + GAMMA * (reflected_vertex - centroid)
            expanded_score = f(expanded_vertex)  # Run trial (~10 seconds)
            
            print(f"     → Expanded: J={expanded_score:.3f}")
            
            # Use whichever is better: expanded or reflected
            if expanded_score < reflected_score:
                print(f"     ✓ Expansion accepted (excellent!)")
                S[-1] = expanded_vertex
                F[-1] = expanded_score
            else:
                print(f"     ✓ Reflection accepted (expansion didn't help)")
                S[-1] = reflected_vertex
                F[-1] = reflected_score
            
            continue  # Go to next iteration
        
        # ====================================================================
        # OPERATION 3: CONTRACTION
        # ====================================================================
        # IDEA: Reflection didn't help. Pull the worst point TOWARD centroid
        # 
        # Visual:
        #   best1 •
        #        / \
        #       /   \
        # best2• C •worst    →    squeeze inward    →   • contracted
        #
        # Math: contracted = centroid + RHO × (worst - centroid)
        #   where RHO=0.5 means "move halfway from worst to centroid"
        #
        # Example: If centroid=[0.15, 0.08, 0.12] and worst=[0.10, 0.05, 0.08]
        #   contracted = [0.15, 0.08, 0.12] + 0.5 × ([0.10, 0.05, 0.08] - [0.15, 0.08, 0.12])
        #              = [0.15, 0.08, 0.12] + 0.5 × [-0.05, -0.03, -0.04]
        #              = [0.15, 0.08, 0.12] + [-0.025, -0.015, -0.02]
        #              = [0.125, 0.065, 0.10]
        
        contracted_vertex = centroid + RHO * (worst_vertex - centroid)
        contracted_score = f(contracted_vertex)  # Run trial (~10 seconds)
        
        print(f"     → Contracted: J={contracted_score:.3f}")
        
        if contracted_score < worst_score:
            # Contraction helped! Use it
            print(f"     ✓ Contraction accepted")
            S[-1] = contracted_vertex
            F[-1] = contracted_score
            continue  # Go to next iteration
        
        # ====================================================================
        # OPERATION 4: SHRINK
        # ====================================================================
        # IDEA: Nothing worked. Shrink entire simplex toward best point
        # 
        # Visual:
        #    before:              after:
        #      •                    •
        #     /|\                  /|\
        #    / | \                / | \
        #   •  •  •     →        •  •  •  (all closer to top)
        #
        # Math: For each vertex i (except best):
        #   vertex[i] = best + SIGMA × (vertex[i] - best)
        #   where SIGMA=0.5 means "move halfway toward best"
        #
        # This is expensive: requires re-evaluating 3 vertices (3 × 10s = 30s)
        
        print(f"     ✗ All moves failed. Shrinking entire simplex toward best point...")
        
        best_vertex = S[0].copy()
        
        # Shrink all vertices except the best one (index 0)
        for i in range(1, len(S)):
            # Move vertex i halfway toward the best vertex
            S[i] = best_vertex + SIGMA * (S[i] - best_vertex)
            
            # Re-evaluate this shrunk vertex
            F[i] = f(S[i])  # Run trial (~10 seconds)
            
            print(f"     → Shrunk vertex {i}: J={F[i]:.3f}")
    
    # ========================================================================
    # STEP 4: Return the best solution found
    # ========================================================================
    # Sort one final time to ensure S[0] is truly the best
    final_indices = np.argsort(F)
    S = S[final_indices]
    F = F[final_indices]
    
    best_gains = S[0]
    best_score = F[0]
    
    print(f"\n[NM] Optimization complete after {iteration+1} iterations")
    print(f"     Final best: Kp={best_gains[0]:.4f} Ki={best_gains[1]:.4f} Kd={best_gains[2]:.4f}")
    print(f"     Final score: J={best_score:.3f}")
    
    return best_gains, float(best_score)

def start_nm_tuning(
    motor_pids,                           # [pid1, pid2, pid3]
    trial_sec=10.0,                       # each trial duration
    w1=1.0, w2=0.7, w3=1.2, pctl=95,
    x0=(0.16, 0.07, 0.14),                # initial (Kp,Ki,Kd)
    scale=0.3,
    max_iter=20,
    gains_lock=None,                      # thread lock for updating globals
    update_globals_fn=None,               # callback to update current_kp/ki/kd
    sync_gui_fn=None                      # callback to sync GUI sliders
):
    """
    Launch tuner in a daemon thread so your GUI/video loop keeps running.
    
    Args:
        gains_lock: Threading lock for thread-safe gain updates
        update_globals_fn: Function to update global gain variables
        sync_gui_fn: Function to sync GUI sliders after optimization
    """
    def worker():
        print("[NM] starting tuner…")
        x0_np = np.array(x0, dtype=float)

        # wrap f(gains) → calls one physical trial to get J
        def f_eval(g: np.ndarray) -> float:
            kp, ki, kd = clip_gains(*g)
            return run_trial(
                pid_list=motor_pids,
                trial_duration_s=trial_sec,
                w1=w1, w2=w2, w3=w3, pctl=pctl,
                gains=(kp, ki, kd)
            )

        best_g, best_J = nelder_mead_minimize(
            f=f_eval,
            x0=x0_np,
            scale=scale,
            max_iter=max_iter,
            tol_f=1e-3,
            tol_x=1e-3
        )
        kp, ki, kd = clip_gains(*best_g)
        print("\n" + "="*70)
        print(f"[NM] ★ OPTIMIZATION COMPLETE ★")
        print(f"[NM] BEST  Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f}   ->   J={best_J:.3f}")
        print("="*70 + "\n")
        apply_gains(motor_pids, kp, ki, kd, reset_integral=False,
                   gains_lock=gains_lock, update_globals_fn=update_globals_fn)
        
        # Sync GUI sliders with optimized values
        if sync_gui_fn is not None:
            try:
                sync_gui_fn()
            except Exception as e:
                print(f"[NM] Warning: Could not sync GUI: {e}")

    threading.Thread(target=worker, daemon=True).start()