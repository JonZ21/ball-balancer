import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_error_series(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    Extract error series. Returns (x, y, mag, x_label, y_label).
    If one axis is missing, returns empty array for that axis.
    If neither x nor y present, tries 'xy_magnitude' for mag, leaving x,y empty.
    """
    columns_lower = {c.lower(): c for c in df.columns}
    x = np.array([])
    y = np.array([])
    x_label = "x"
    y_label = "y"

    if "x" in columns_lower:
        cx = columns_lower["x"]
        x = df[cx].to_numpy(dtype=float)
        x_label = cx
    if "y" in columns_lower:
        cy = columns_lower["y"]
        y = df[cy].to_numpy(dtype=float)
        y_label = cy

    if x.size and y.size:
        mag = np.sqrt(x ** 2 + y ** 2)
    else:
        if "xy_magnitude" in columns_lower:
            cm = columns_lower["xy_magnitude"]
            mag = df[cm].to_numpy(dtype=float)
        else:
            # If only one axis exists, magnitude = |that axis|
            if x.size:
                mag = np.abs(x)
            elif y.size:
                mag = np.abs(y)
            else:
                raise ValueError("No suitable error columns found. Need 'x'/'y' or 'xy_magnitude'.")

    return x, y, mag, x_label, y_label


def compute_error_metrics_vector(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    window_frac: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    For vector error e(t) = [x(t), y(t)]:
      - Initial error vector e0 = median over first window (component-wise).
      - Define signed projection s(t) = dot(e(t), unit(e0)).
      - Overshoot (%) = max(0, -min_t s(t)) / ||e0|| * 100  (how far past zero along e0 direction)
      - Accuracy (steady-state MAE) = mean(||e(t)||) over last window.

    Returns: overshoot_pct, accuracy_mae_mag, e0_norm, min_projection
    """
    n = len(t)
    if n == 0:
        raise ValueError("Empty series.")
    if x.size == 0 or y.size == 0:
        raise ValueError("Vector metrics require both x and y.")

    window = max(5, int(max(1, window_frac * n)))
    e0 = np.array([np.median(x[:window]), np.median(y[:window])], dtype=float)
    e0_norm = float(np.linalg.norm(e0))
    if e0_norm < 1e-12:
        # If the initial error is ~0, overshoot not meaningful
        overshoot_pct = 0.0
    else:
        e0_unit = e0 / e0_norm
        s = x * e0_unit[0] + y * e0_unit[1]
        min_projection = float(np.min(s))
        overshoot_along = max(0.0, -min_projection)  # only count going past zero along the initial direction
        overshoot_pct = (overshoot_along / e0_norm) * 100.0

    mag = np.sqrt(x ** 2 + y ** 2)
    accuracy_mae_mag = float(np.mean(np.abs(mag[-window:])))
    min_projection = float("nan")
    if e0_norm >= 1e-12:
        e0_unit = e0 / e0_norm
        s = x * e0_unit[0] + y * e0_unit[1]
        min_projection = float(np.min(s))

    return overshoot_pct, accuracy_mae_mag, e0_norm, min_projection


def compute_error_metrics_scalar(
    t: np.ndarray,
    e: np.ndarray,
    window_frac: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    For scalar error e(t):
      - e0 = median over first window.
      - s0 = sign(e0). Overshoot happens when e crosses zero and goes opposite sign.
      - Overshoot (%) = max_t max(0, -s0*e(t)) / |e0| * 100.
      - Accuracy (steady-state MAE) = mean(|e(t)|) over last window.

    Returns: overshoot_pct, accuracy_mae_abs, e0_abs, min_signed
    """
    n = len(e)
    if n == 0:
        raise ValueError("Empty series.")
    window = max(5, int(max(1, window_frac * n)))
    e0 = float(np.median(e[:window]))
    e0_abs = abs(e0)

    if e0_abs < 1e-12:
        overshoot_pct = 0.0
        min_signed = float("nan")
    else:
        s0 = 1.0 if e0 >= 0 else -1.0
        signed = s0 * e  # zero crossing occurs when this becomes negative
        min_signed = float(np.min(signed))
        overshoot_amount = max(0.0, -min_signed)
        overshoot_pct = (overshoot_amount / e0_abs) * 100.0

    accuracy_mae_abs = float(np.mean(np.abs(e[-window:])))
    return overshoot_pct, accuracy_mae_abs, e0_abs, min_signed


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV skipping comment lines (starting with '#') and malformed rows.
    Ensures 'time' is present and numeric.
    """
    df = pd.read_csv(
        path,
        comment="#",
        on_bad_lines="skip",  # pandas >= 1.3
    )
    # Normalize column names just for accessing; keep originals for plotting labels
    renamed = {c: c.strip() for c in df.columns}
    df = df.rename(columns=renamed)

    # Require a 'time' column (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if "time" not in cols_lower:
        raise ValueError("CSV is missing required 'time' column.")
    time_col = cols_lower["time"]
    df = df.dropna(subset=[time_col])
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)
    return df


def analyze_and_plot(csv_path: str, save: bool = True, return_metrics: bool = False):
    df = load_csv(csv_path)

    # Identify time and error series
    time_col = {c.lower(): c for c in df.columns}["time"]
    x, y, mag, x_label, y_label = extract_error_series(df)
    time_raw = df[time_col].to_numpy(dtype=float)

    # Offset time to start from zero
    t0 = time_raw - time_raw[0]

    # Compute metrics
    if x.size and y.size:
        overshoot_pct, accuracy_mae, e0_norm, min_projection = compute_error_metrics_vector(t0, x, y)
        # Plot: x, y, magnitude
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t0, x, label=x_label, color="#1f77b4")
        axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        axes[0].set_ylabel("x (pixels)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t0, y, label=y_label, color="#ff7f0e")
        axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        axes[1].set_ylabel("y (pixels)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(t0, mag, label="|error|", color="#2ca02c")
        axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        axes[2].set_ylabel("|Error| (pixels)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        # textbox = (
        #     f"Overshoot (along initial error): {overshoot_pct:.2f}%\n"
        #     f"Accuracy (steady-state |error| MAE): {accuracy_mae:.4f}\n"
        #     f"|e0|: {e0_norm:.4f}"
        # )
        # axes[0].text(
        #     0.02,
        #     0.98,
        #     textbox,
        #     transform=axes[0].transAxes,
        #     verticalalignment="top",
        #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        #     fontsize=10,
        # )
        fig.suptitle("Error Response Analysis (vector)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        # Scalar case: use whichever is available (x or y or mag)
        if x.size:
            series = x
            label = x_label
            label = "x (pixels)"
        elif y.size:
            series = y
            # label = y_label
            label = "y (pixels)"
        else:
            series = mag
            label = "|Error|(pixels)"

        overshoot_pct, accuracy_mae, e0_abs, min_signed = compute_error_metrics_scalar(t0, series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t0, series, label=label, color="#1f77b4")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title("Error Response Analysis (scalar)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        textbox = (
            f"Overshoot: {overshoot_pct:.2f}%\n"
            f"Accuracy (steady-state MAE): {accuracy_mae:.4f}\n"
            f"|e0|: {e0_abs:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            textbox,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )
        fig.tight_layout()

    # Save or show
    out_path = ""
    if save:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(os.path.dirname(csv_path) or ".", f"analysis_{base}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    # Print concise results to stdout
    print(f"File: {csv_path}")
    print(f"Overshoot: {overshoot_pct:.2f}%")
    print(f"Accuracy (steady-state MAE): {accuracy_mae:.6f}")
    if out_path:
        print(f"Plot saved to: {out_path}")

    if return_metrics:
        return out_path, float(overshoot_pct), float(accuracy_mae)
    return out_path


def analyze_folder(folder_path: str, save: bool = True) -> None:
    """
    Analyze all CSV files in a folder, printing per-file results and the averages.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    csv_files = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if fname.lower().endswith(".csv")
    ]

    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return

    overshoots: list[float] = []
    accuracies: list[float] = []

    for csv_path in csv_files:
        try:
            _, o, a = analyze_and_plot(csv_path, save=save, return_metrics=True)
            if np.isfinite(o):
                overshoots.append(float(o))
            if np.isfinite(a):
                accuracies.append(float(a))
        except Exception as exc:
            print(f"Skipping {csv_path}: {exc}")

    if overshoots and accuracies:
        avg_overshoot = float(np.mean(overshoots))
        avg_accuracy = float(np.mean(accuracies))
        print(f"Analyzed {len(overshoots)} CSV files in folder: {folder_path}")
        print(f"Average Overshoot: {avg_overshoot:.2f}%")
        print(f"Average Accuracy (steady-state MAE): {avg_accuracy:.6f}")
    else:
        print(f"No valid results computed from CSVs in: {folder_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze response overshoot and accuracy from CSV logs.")
    parser.add_argument("csv", nargs="?", help="Path to CSV file (with optional '#' header comment lines).")
    parser.add_argument(
        "--folder",
        help="Path to a folder containing CSV files. Computes per-file metrics and their averages.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save plot, display it instead.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.folder:
        analyze_folder(args.folder, save=not args.no_save)
    elif args.csv:
        analyze_and_plot(args.csv, save=not args.no_save)
    else:
        raise SystemExit("Provide a CSV path or use --folder to analyze a directory.")

