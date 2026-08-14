#!/usr/bin/env python3
"""
Beam stabilisation / actuator characterisation
+ Status overlay, error vectors, mouse targets, help, emergency stop
+ Jacobian response plots
+ Open-loop vs Closed-loop time series + PSD analysis
+ Live detection trackbars
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union
from collections import deque
import json
import glob
import os

import numpy as np
import cv2
from scipy import ndimage, signal
from pprint import pprint
import matplotlib.pyplot as plt

import pylablib as pll
from pylablib.devices import Thorlabs, Newport
# =====================================================================
# 1. Configuration
# =====================================================================

@dataclass(frozen=True)
class CameraConfig:
    roi: Tuple[int, int, int, int] = (0, 1440, 0, 1080)
    exposure: float = 0.02
    frame_period: float = 0.035

@dataclass
class DetectionConfig:          # now mutable so trackbars can change it
    min_spot_area: int = 50
    threshold: int = 3
    gamma: float = 1.1
    contrast_alpha: float = 4.0
    contrast_beta: float = 20
    display_alpha: float = 5.0
    display_beta: float = 50

@dataclass(frozen=True)
class CharacterisationConfig:
    axes: Tuple[int, ...] = (1, 2, 3, 4)
    step_sizes = tuple(range(-40, 41, 10))
    settle_frames: int = 15

# =====================================================================
# 2. Narrow command interface
# =====================================================================

Point = Tuple[float, float]

@dataclass
class MoveAxis:
    axis: int
    steps: int

@dataclass
class SetTarget:
    spot: int
    position: Point

@dataclass
class SpotError:
    dx: float
    dy: float
    dist: float

@dataclass
class Finished:
    reason: str = ""

Command = Union[MoveAxis, SetTarget, Finished]

# =====================================================================
# 3. Pure vision pipeline
# =====================================================================

def preprocess_image(gray: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    background = np.mean(gray)
    gray = gray.astype(np.float32) - background
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = (gray / 255.0) ** cfg.gamma
    gray = (gray * 255).astype(np.uint8)
    return gray


def detect_two_spots(gray: np.ndarray, cfg: DetectionConfig) -> List[Tuple]:
    _, binary = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    spots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > cfg.min_spot_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            spots.append((x, y, w, h, area))

    spots = sorted(spots, key=lambda s: s[4], reverse=True)[:2]
    return spots


def calculate_centroid(gray: np.ndarray, roi: Tuple, cfg: DetectionConfig) -> Optional[Point]:
    x, y, w, h = roi
    roi_img = gray[y:y+h, x:x+w].astype(np.float32)
    roi_img[roi_img < cfg.threshold] = 0
    if np.sum(roi_img) == 0:
        return None
    cy, cx = ndimage.center_of_mass(roi_img)
    return (cx + x, cy + y)


def process_frame(frame: np.ndarray, cfg: DetectionConfig) -> Tuple[Optional[List[Point]], np.ndarray]:
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(frame).squeeze()

    if gray.dtype == np.uint16:
        gray = (gray >> 8).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    gray = cv2.convertScaleAbs(gray, alpha=cfg.contrast_alpha, beta=cfg.contrast_beta)
    processed = preprocess_image(gray, cfg)

    spots = detect_two_spots(processed, cfg)
    if len(spots) != 2:
        return None, processed

    centroids = []
    for spot in spots:
        x, y, w, h, _ = spot
        c = calculate_centroid(processed, (x, y, w, h), cfg)
        if c is not None:
            centroids.append(c)

    if len(centroids) != 2:
        return None, processed
    return centroids, processed

# =====================================================================
# 4. TargetSetter
# =====================================================================

class TargetSetter:
    def __init__(self, move_step: int = 8):
        self.move_step = move_step
        self.active = False
        self.current_spot = 0
        self.targets: Dict[int, Optional[Point]] = {0: None, 1: None}
        self._image_size: Tuple[int, int] = (1440, 1080)

    def start(self, centroids: Optional[List[Point]], image_size: Tuple[int, int]):
        self.active = True
        self.current_spot = 0
        self._image_size = image_size
        w, h = image_size
        centre = (w / 2.0, h / 2.0)
        self.targets[0] = centroids[0] if centroids and len(centroids) > 0 else centre
        self.targets[1] = centroids[1] if centroids and len(centroids) > 1 else centre
        print("\n=== Target Setting Mode === (mouse click + WASD)")

    def set_target_from_mouse(self, x: float, y: float):
        if not self.active:
            return
        w, h = self._image_size
        x = max(0.0, min(w - 1.0, x))
        y = max(0.0, min(h - 1.0, y))
        self.targets[self.current_spot] = (x, y)
        print(f"Mouse → Target {self.current_spot+1}: ({x:.1f}, {y:.1f})")

    def handle(self, key: int, centroids: Optional[List[Point]]) -> Tuple[bool, List[Command]]:
        if not self.active:
            return False, []

        commands: List[Command] = []
        tx, ty = self.targets[self.current_spot]
        moved = False

        if key in (82, ord('w')):
            ty -= self.move_step; moved = True
        elif key in (84, ord('s')):
            ty += self.move_step; moved = True
        elif key in (81, ord('a')):
            tx -= self.move_step; moved = True
        elif key in (83, ord('d')):
            tx += self.move_step; moved = True
        elif key == ord('n'):
            self.current_spot = 1 - self.current_spot
            print(f"Now adjusting Target {self.current_spot + 1}")
        elif key == 13:
            self.active = False
            print("Both targets locked")
            for i, pos in self.targets.items():
                if pos is not None:
                    print(f"  Target {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
            return False, []

        if moved:
            w, h = self._image_size
            tx = max(0.0, min(w - 1.0, tx))
            ty = max(0.0, min(h - 1.0, ty))
            self.targets[self.current_spot] = (tx, ty)
            commands.append(SetTarget(self.current_spot, (tx, ty)))

        return self.active, commands

    def get_targets(self) -> Dict[int, Optional[Point]]:
        return self.targets.copy()

    def stop(self):
        if self.active:
            self.active = False
            print("Target setting cancelled")

# =====================================================================
# 5. Characteriser (with plot method)
# =====================================================================

class Characteriser:
    def __init__(self, cfg: CharacterisationConfig):
        self.cfg = cfg
        self.active = False
        self.axis_idx = 0
        self.step_idx = 0
        self.start_pos: Optional[List[Point]] = None
        self.results: Dict[int, List[dict]] = {ax: [] for ax in cfg.axes}
        self.jacobian = np.zeros((4, 4))
        self.pending_move: Optional[MoveAxis] = None
        self.settle_countdown = 0
        self.abs_pos = 0
        self.control_matrix = None

    def start(self):
        self.active = True
        self.axis_idx = 0
        self.step_idx = 0
        self.start_pos = None
        self.results = {ax: [] for ax in self.cfg.axes}
        self.jacobian = np.zeros((4, 4))
        self.control_matrix = None
        self.pending_move = None
        self.settle_countdown = 0
        self.abs_pos = 0
        print("\n=== Starting 4-Axis Characterisation ===")

    def handle(self, centroids: Optional[List[Point]]) -> Tuple[bool, List[Command]]:
        if not self.active:
            return False, []

        commands: List[Command] = []
        axes = self.cfg.axes
        steps = self.cfg.step_sizes

        if self.settle_countdown > 0:
            self.settle_countdown -= 1
            return True, []

        if self.pending_move is not None and self.start_pos is not None:
            if centroids and len(centroids) >= 2:
                dx1 = centroids[0][0] - self.start_pos[0][0]
                dy1 = centroids[0][1] - self.start_pos[0][1]
                dx2 = centroids[1][0] - self.start_pos[1][0]
                dy2 = centroids[1][1] - self.start_pos[1][1]
                ax = self.pending_move.axis
                self.results[ax].append({
                    "abs_pos": self.abs_pos,
                    "dx1": dx1, "dy1": dy1, "dx2": dx2, "dy2": dy2,
                })
                print(f"  Axis {ax} | abs={self.abs_pos:5d}  "
                      f"Δx1={dx1:7.2f} Δy1={dy1:7.2f}  Δx2={dx2:7.2f} Δy2={dy2:7.2f}")
            self.pending_move = None

        if self.step_idx >= len(steps):
            self._fit_column(axes[self.axis_idx])
            self.axis_idx += 1
            self.step_idx = 0
            self.start_pos = None
            self.abs_pos = 0

            if self.axis_idx >= len(axes):
                self.active = False
                print("\n=== Characterisation Complete ===")
                print(self.jacobian)
                self._save()
                self.plot_responses()          # auto-plot
                return False, [Finished("characterisation")]

        if self.step_idx == 0 and self.start_pos is None:
            if not centroids or len(centroids) < 2:
                print(f"Axis {axes[self.axis_idx]}: no beam – skipping")
                self.axis_idx += 1
                if self.axis_idx >= len(axes):
                    self.active = False
                    return False, [Finished("characterisation")]
                return True, []
            self.start_pos = [centroids[0], centroids[1]]
            print(f"Axis {axes[self.axis_idx]} start: "
                  f"Spot1=({self.start_pos[0][0]:.1f},{self.start_pos[0][1]:.1f})  "
                  f"Spot2=({self.start_pos[1][0]:.1f},{self.start_pos[1][1]:.1f})")

        if self.start_pos is not None and self.pending_move is None:
            rel = steps[self.step_idx]
            ax = axes[self.axis_idx]
            commands.append(MoveAxis(ax, rel))
            self.pending_move = MoveAxis(ax, rel)
            self.abs_pos += rel
            self.settle_countdown = self.cfg.settle_frames
            self.step_idx += 1

        return self.active, commands

    def _fit_column(self, axis: int):
        data = self.results[axis]
        if len(data) < 2:
            print(f"Axis {axis}: not enough points")
            return
        steps_arr = np.array([r["abs_pos"] for r in data], dtype=float)
        channels = {
            "dx1": np.array([r["dx1"] for r in data], dtype=float),
            "dy1": np.array([r["dy1"] for r in data], dtype=float),
            "dx2": np.array([r["dx2"] for r in data], dtype=float),
            "dy2": np.array([r["dy2"] for r in data], dtype=float),
        }
        col = axis - 1
        slopes = []
        print(f"\n--- Axis {axis} linearity check ---")
        for name, measured in channels.items():
            slope, intercept = np.polyfit(steps_arr, measured, 1)
            residual = measured - (slope * steps_arr + intercept)
            rms = np.sqrt(np.mean(residual**2))
            print(f"  {name}: slope = {slope:8.4f}   residual RMS = {rms:5.2f} px")
            slopes.append(slope)
        self.jacobian[:, col] = slopes
        print(f"Axis {axis} Jacobian column: {slopes}")

    def compute_control_matrix(self, lambda_reg=0.5):
        J = self.jacobian
        n = J.shape[1]
        self.M = np.linalg.inv(J.T @ J + lambda_reg * np.eye(n)) @ J.T
        return self.M

    def _save(self):
        self.control_matrix = self.compute_control_matrix(0.5)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": timestamp,
            "description": "Picomotor 8742 actuator characterisation – pixels per step",
            "results": self.results,
            "jacobian": self.jacobian.tolist(),
            "control_matrix": self.control_matrix.tolist(),
        }
        filename = f"actuator_characterization_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Results saved to: {filename}")

    def plot_responses(self):
        """Open a matplotlib window with the four response curves + Jacobian."""
        if not any(self.results.values()):
            print("No characterisation data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("Actuator Response Curves (pixels vs motor steps)", fontsize=13)
        channel_names = ["dx1", "dy1", "dx2", "dy2"]
        colors = ["C0", "C1", "C2", "C3"]

        for ax_idx, ax_num in enumerate(self.cfg.axes):
            data = self.results.get(ax_num, [])
            if len(data) < 2:
                continue
            row, col = divmod(ax_idx, 2)
            ax = axes[row, col]
            steps = np.array([r["abs_pos"] for r in data])

            for ch, color in zip(channel_names, colors):
                y = np.array([r[ch] for r in data])
                ax.plot(steps, y, "o-", color=color, label=ch, markersize=4)

            ax.set_title(f"Axis {ax_num}")
            ax.set_xlabel("Motor steps (abs)")
            ax.set_ylabel("Δ pixels")
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=8)

        # Add Jacobian text
        j_str = "Jacobian (px/step):\n" + np.array2string(self.jacobian, precision=3, suppress_small=True)
        fig.text(0.02, 0.02, j_str, family="monospace", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show(block=False)
        print("Response curves plotted (close window when finished looking)")

    def stop(self):
        if self.active:
            self.active = False
            self.pending_move = None
            self.settle_countdown = 0
            print("\n=== Characterisation aborted ===")

    def status_text(self) -> str:
        if not self.active:
            return ""
        n_steps = len(self.cfg.step_sizes)
        return (f"Axis {self.axis_idx+1}/{len(self.cfg.axes)}  "
                f"Step {min(self.step_idx, n_steps)}/{n_steps}  "
                f"Settle {self.settle_countdown}")

from collections import deque

class ExponentialFilter:
    def __init__(self, alpha: float, median_len: int):
        self.alpha = alpha
        self.median_len = median_len
        self.buffer = deque(maxlen=median_len)
        self.y = None

    def update(self, x: float) -> float:
        # short median
        self.buffer.append(x)
        if len(self.buffer) == self.median_len:
            x = float(np.median(self.buffer))

        # exponential smoothing
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y
# =====================================================================
# 6. ClosedLoop + BeamJog
# =====================================================================

class ClosedLoop:
    def __init__(self, characteriser):
        self.characteriser = characteriser
        self.M: Optional[np.ndarray] = None
        self.active = False
        self.last_commands: Optional[np.ndarray] = None
        self.last_steps: List[int] = [0, 0, 0, 0]
        self.deadband = 0.5
        self.gain = 1.5

        # error filters (outer loop)
        self.filt_x1 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_x2 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_y1 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_y2 = ExponentialFilter(alpha=0.15, median_len=3)

        # absolute centroid position filters (inner loop)
        self.filt_cx1 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_cy1 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_cx2 = ExponentialFilter(alpha=0.15, median_len=3)
        self.filt_cy2 = ExponentialFilter(alpha=0.15, median_len=3)

        # closed-loop plant model
        self.J_hat = None
        self.P = None # covariance
        self.lambda_forget = 0.99
        self.min_move = 2.0 
        # only update model if 
        # actuator move is greater than min_move (px)
        # for forming Δu and Δx
        self.prev_filtered_pos: Optional[np.ndarray] = None
        self.pending_delta_u: Optional[np.ndarray] = None
        self.prev_filtered_error: Optional[np.ndarray] = None

        self.debug_print = False
        self.disable_filters = False # disable error filters for diagnosis
        self.disable_rls = True # start with adaptation OFF for diagnosis
        self.diag_gain = None # override gain when not None
        self.diag_deadband = None # override deadband when not None

    def start(self):
        # initialise regularised inverse matrix
        if self.characteriser.control_matrix is not None:
            self.M = self.characteriser.control_matrix.copy()
        else:
            self.M = self.characteriser.compute_control_matrix(0.5)
        if self.M is None:
            raise RuntimeError("No control matrix – run characterisation first")

        # initialise forward sensitivty matrix
        # for plant model from Jacobian
        self.J_hat = self.characteriser.jacobian.copy()

        # covariance
        self.P = 1e3 * np.eye(4)
        self.M *= self.gain

        self.pending_delta_u = None
        self.prev_filtered_error = None
        self.prev_filtered_pos = None
        self.active = True
        print(f"Closed-loop started (gain={self.gain:.2f}, deadband={self.deadband:.2f})")

    def _update_plant_model(self, delta_u, delta_x):
        """Recursive least-squares update of J_hat using latest move"""
        if np.linalg.norm(delta_u) < self.min_move:
            return

        # residual
        e = delta_x - self.J_hat @ delta_u

        # Kalman gain
        Pu = self.P @ delta_u
        denom = self.lambda_forget + delta_u @ Pu
        if abs(denom) < 1e-12:
            return
        K = Pu / denom

        # update Jacobian
        # TODO: what does matrix transform np.outer do?
        self.J_hat = self.J_hat + np.outer(e, K)

        # update covariance
        self.P = (self.P - np.outer(K, delta_u) @ self.P) / self.lambda_forget

        # optional: keep P from becoming too small / negative
        self.P = 0.5 * (self.P + self.P.T) # enforce symmetry and avoid P from becoming too small
        self.P += 1e-6 * np.eye(4) # small regularisation to keep safely positive

        # recompute control matrix from the new J_hat
        lambda_reg = 0.5
        n = self.J_hat.shape[1]
        self.M = np.linalg.inv(self.J_hat.T @ self.J_hat + lambda_reg * np.eye(n)) @ self.J_hat.T
        self.M *= self.gain

    def compute_commands(self, errors: Dict[int, SpotError]) -> np.ndarray:
        """Returns (command vector a, filtered error vector e_filt)"""
        dx1 = self.filt_x1.update(errors[0].dx)
        dx2 = self.filt_x2.update(errors[1].dx)
        dy1 = self.filt_y1.update(errors[0].dy)
        dy2 = self.filt_y2.update(errors[1].dy)
        self.e_filt = np.array([dx1, dy1, dx2, dy2], dtype=float)
        self.a = -self.M @ self.e_filt
        self.last_commands = self.a
        return self.a, self.e_filt

    def postprocess(self, a: np.ndarray, max_steps: int = 200) -> List[int]:
        steps = []
        for val in a:
            if abs(val) < self.deadband:
                steps.append(0)
            else:
                s = int(round(val))
                s = max(-max_steps, min(max_steps, s))
                steps.append(s)
        return steps

    def handle(self, errors: Dict[int, SpotError],
            centroids: Optional[List[Point]] = None) -> Tuple[bool, List]:
        if not self.active or self.M is None:
            return False, []

        # Outer loop: filter errors and compute commands
        a, e_filt = self.compute_commands(errors)
        steps = self.postprocess(a)
        self.last_steps = steps
        commands = [MoveAxis(i + 1, s) for i, s in enumerate(steps) if s != 0]

        # Filter absolute centroids once (if available)
        curr = None
        if centroids is not None and len(centroids) == 2:
            cx1 = self.filt_cx1.update(centroids[0][0])
            cy1 = self.filt_cy1.update(centroids[0][1])
            cx2 = self.filt_cx2.update(centroids[1][0])
            cy2 = self.filt_cy2.update(centroids[1][1])
            curr = np.array([cx1, cy1, cx2, cy2], dtype=float)

        # ----- Inner loop: update plant model when we have a response -----
        if (curr is not None
                and self.pending_delta_u is not None
                and self.prev_filtered_pos is not None):

            delta_x = curr - self.prev_filtered_pos
            self._update_plant_model(self.pending_delta_u, delta_x)
            self.pending_delta_u = None          # consumed

        # Store the command we are about to apply so we can update the model later
        if any(s != 0 for s in steps):
            self.pending_delta_u = np.array(steps, dtype=float)
        # Keep previous filtered position for next Δx
        if curr is not None:
            self.prev_filtered_pos = curr

        return True, commands

    def stop(self):
        if self.active:
            self.active = False
            print("Closed-loop stopped")

class BeamJog:
    def __init__(self, characteriser, pixel_step: float = 2000.0):
        self.characteriser = characteriser
        self.active = False
        self.pixel_step = pixel_step
        self.current_spot = 0
        self.last_steps: List[int] = [0, 0, 0, 0]

    def start(self):
        if self.characteriser.control_matrix is None:
            print("No control matrix – run characterisation first")
            return
        self.active = True
        print("=== Beam Jog Mode ===")

    def stop(self):
        if self.active:
            self.active = False
            print("Beam jog stopped")

    def handle(self, key: int) -> Tuple[bool, List[Command]]:
        if not self.active:
            return False, []
        if key == 13:
            self.stop()
            return False, []
        if key == ord('n'):
            self.current_spot = 1 - self.current_spot
            print(f"Now moving Spot {self.current_spot + 1}")
            return True, []

        dx = dy = 0.0
        if key in (82, ord('w')): dy = +self.pixel_step
        elif key in (84, ord('s')): dy = -self.pixel_step
        elif key in (81, ord('a')): dx = +self.pixel_step
        elif key in (83, ord('d')): dx = -self.pixel_step
        if dx == 0 and dy == 0:
            return True, []

        e = np.zeros(4)
        if self.current_spot == 0:
            e[0], e[1] = dx, dy
        else:
            e[2], e[3] = dx, dy

        a = -self.characteriser.control_matrix @ e
        steps = [int(round(v)) for v in a]
        self.last_steps = steps
        commands = [MoveAxis(i+1, s) for i, s in enumerate(steps) if s != 0]
        print(f"Spot {self.current_spot+1}: Δpix=({dx:+.0f},{dy:+.0f}) → {steps}")
        return True, commands

# =====================================================================
# 7. Data Recorder + PSD / Time-series analysis
# =====================================================================

class DataRecorder:
    """Records centroid time series for open-loop vs closed-loop comparison."""

    def __init__(self):
        self.active = False
        self.label = "open"          # "open" or "closed"
        self.start_t = None
        self.data: List[Tuple[float, float, float, float, float]] = []  # t, x1,y1,x2,y2
        self.last_open: Optional[dict] = None
        self.last_closed: Optional[dict] = None

    def start(self, is_closed_loop: bool):
        self.active = True
        self.label = "closed" if is_closed_loop else "open"
        self.start_t = time.perf_counter()
        self.data = []
        print(f"\n▶ Recording {self.label}-loop … press 'r' again to stop")

    def update(self, centroids: Optional[List[Point]]):
        if not self.active or centroids is None or len(centroids) < 2:
            return
        t = time.perf_counter() - self.start_t
        x1, y1 = centroids[0]
        x2, y2 = centroids[1]
        self.data.append((t, x1, y1, x2, y2))

    def stop(self):
        if not self.active:
            return
        self.active = False
        duration = self.data[-1][0] if self.data else 0.0
        n = len(self.data)
        print(f"■ Recording stopped ({self.label}-loop): {n} samples, {duration:.1f}s")

        if n < 10:
            print("Too few samples – discarded")
            return

        arr = np.array(self.data)
        t = arr[:, 0]
        pos = arr[:, 1:]          # N x 4

        # Simple RMS of each channel (relative to mean)
        rms = np.std(pos, axis=0)
        print(f"  RMS  x1={rms[0]:.2f}  y1={rms[1]:.2f}  x2={rms[2]:.2f}  y2={rms[3]:.2f} px")

        record = {
            "label": self.label,
            "t": t,
            "pos": pos,             # columns: x1 y1 x2 y2
            "rms": rms,
            "fs": (n - 1) / duration if duration > 0 else 30.0,
        }

        if self.label == "open":
            self.last_open = record
        else:
            self.last_closed = record

        # Auto-save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"beam_record_{self.label}_{ts}.npz"
        np.savez(fname, t=t, pos=pos, rms=rms, label=self.label)
        print(f"  Saved → {fname}")

    def plot_comparison(self):
        """Plot time series + PSD for last open vs last closed."""
        if self.last_open is None and self.last_closed is None:
            print("No recordings yet. Use 'r' to record open and closed loop data.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Open-loop vs Closed-loop Beam Stability", fontsize=14)

        channels = ["x1", "y1", "x2", "y2"]
        colors = {"open": "C0", "closed": "C1"}

        for i, ch in enumerate(channels):
            ax = axes[i // 2, i % 2]

            for rec, style in [(self.last_open, "-"), (self.last_closed, "-")]:
                if rec is None:
                    continue
                t = rec["t"]
                y = rec["pos"][:, i]
                # detrend for display
                y = y - np.mean(y)
                ax.plot(t, y, style, color=colors[rec["label"]],
                        label=f"{rec['label']} (RMS={rec['rms'][i]:.2f}px)", alpha=0.85)

            ax.set_title(ch)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (px, mean removed)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show(block=False)

        # ----- PSD figure -----
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        fig2.suptitle("Power Spectral Density (Welch)", fontsize=14)

        for i, ch in enumerate(channels):
            ax = axes2[i // 2, i % 2]

            for rec in (self.last_open, self.last_closed):
                if rec is None:
                    continue
                y = rec["pos"][:, i] - np.mean(rec["pos"][:, i])
                fs = rec["fs"]
                f, Pxx = signal.welch(y, fs=fs, nperseg=min(1024, len(y)//2))
                ax.semilogy(f, Pxx, color=colors[rec["label"]],
                            label=rec["label"], alpha=0.85)

            ax.set_title(ch)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (px²/Hz)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(left=0.1)

        plt.tight_layout()
        plt.show(block=False)
        print("Time-series + PSD plots opened")

    def has_data(self) -> bool:
        return self.last_open is not None or self.last_closed is not None

    def plot_analysis(self, mode: str = "both"):
        """
        mode: "open" | "closed" | "both"
        Opens matplotlib windows for the requested data.
        """
        import matplotlib.pyplot as plt
        from scipy import signal

        records = []
        if mode in ("open", "both") and self.last_open is not None:
            records.append(self.last_open)
        if mode in ("closed", "both") and self.last_closed is not None:
            records.append(self.last_closed)

        if not records:
            print("No recordings in memory. Press 'r' to record first.")
            return

        channels = ["x1", "y1", "x2", "y2"]
        colors = {"open": "C0", "closed": "C1"}

        # ----- Time series -----
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        fig1.suptitle("Position vs Time (mean-removed)", fontsize=13)

        for i, ax in enumerate(axes1.flat):
            for rec in records:
                y = rec["pos"][:, i] - np.mean(rec["pos"][:, i])
                ax.plot(rec["t"], y,
                        label=f"{rec['label']}  RMS={rec['rms'][i]:.2f}px",
                        color=colors[rec["label"]], alpha=0.85, lw=0.9)
            ax.set_title(channels[i])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylabel("px")
        axes1[1, 0].set_xlabel("Time (s)")
        axes1[1, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show(block=False)

        # ----- PSD -----
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
        fig2.suptitle("Power Spectral Density (Welch)", fontsize=13)

        for i, ax in enumerate(axes2.flat):
            for rec in records:
                y = rec["pos"][:, i] - np.mean(rec["pos"][:, i])
                f, Pxx = signal.welch(y, fs=rec["fs"],
                                      nperseg=min(1024, len(y)//2))
                ax.semilogy(f, Pxx, label=rec["label"],
                            color=colors[rec["label"]], alpha=0.85)
            ax.set_title(channels[i])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (px²/Hz)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(left=0.05)

        plt.tight_layout()
        plt.show(block=False)

        # Console summary
        print("\n===== Analysis Summary =====")
        for rec in records:
            print(f"{rec['label']:6s}  RMS: {rec['rms']}")
        if self.last_open is not None and self.last_closed is not None:
            ratio = self.last_open["rms"] / self.last_closed["rms"]
            print(f"Improvement factor (open/closed): {ratio}")
        print("Close the plot windows when finished, then press 'a' or Esc to return to live view.\n")

# =====================================================================
# 8. Display helper
# =====================================================================

def build_display(processed: np.ndarray,
                  centroids: Optional[List[Point]],
                  targets: Dict[int, Optional[Point]],
                  active_spot: Optional[int],
                  cfg: DetectionConfig,
                  mode: str = "IDLE",
                  errors: Optional[Dict[int, SpotError]] = None,
                  char_status: str = "",
                  last_steps: Optional[List[int]] = None,
                  show_help: bool = False,
                  recording: bool = False,
                  rec_label: str = "",
                  scale: float = 0.6) -> np.ndarray:

    display = cv2.convertScaleAbs(processed, alpha=cfg.display_alpha, beta=cfg.display_beta)
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    h, w = display.shape[:2]
    scaled = cv2.resize(display, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = scaled.shape[:2]

    def to_scaled(pt: Point) -> Tuple[int, int]:
        return (int(pt[0] * scale), int(pt[1] * scale))

    if centroids and len(centroids) == 2:
        for i, (cx, cy) in enumerate(centroids):
            p = to_scaled((cx, cy))
            cv2.circle(scaled, p, 8, (0, 255, 0), -1)
            cv2.putText(scaled, f"S{i+1}", (p[0] + 12, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if errors and i in errors:
                dist = errors[i].dist
                color = (0, 255, 0) if dist < 5 else (0, 200, 255) if dist < 20 else (0, 0, 255)
                cv2.putText(scaled, f"{dist:.1f}", (p[0] + 12, p[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for i, pos in targets.items():
        if pos is None:
            continue
        p = to_scaled(pos)
        radius, color = (14, (0, 255, 180)) if (active_spot is not None and i == active_spot) else (10, (0, 200, 0))
        cv2.circle(scaled, p, radius, color, -1)
        cv2.putText(scaled, f"T{i+1}", (p[0] + 16, p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if centroids and len(centroids) == 2 and errors and i in errors:
            c = to_scaled(centroids[i])
            dist = errors[i].dist
            line_color = (0, 255, 0) if dist < 5 else (0, 200, 255) if dist < 20 else (0, 80, 255)
            cv2.line(scaled, p, c, line_color, 2)

    # Status bar
    bar_h = 70
    cv2.rectangle(scaled, (0, sh - bar_h), (sw, sh), (25, 25, 25), -1)

    mode_color = {"IDLE": (180,180,180), "TARGET": (0,255,255), "CHAR": (0,165,255),
                  "LOCK": (0,255,0), "JOG": (255,180,0)}.get(mode, (200,200,200))
    cv2.putText(scaled, f"MODE: {mode}", (12, sh - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

    if recording:
        cv2.putText(scaled, f"● REC {rec_label}", (180, sh - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if char_status:
        cv2.putText(scaled, char_status, (320, sh - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    elif errors and len(errors) == 2:
        e0, e1 = errors.get(0), errors.get(1)
        if e0 and e1:
            cv2.putText(scaled, f"S1 {e0.dist:5.1f}px  S2 {e1.dist:5.1f}px",
                        (320, sh - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    if last_steps is not None:
        steps_str = " ".join(f"{s:+d}" for s in last_steps)
        cv2.putText(scaled, f"steps [{steps_str}]", (12, sh - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    keys = {
        "TARGET": "WASD/mouse  n:switch  Enter:lock  Space:abort",
        "CHAR":   "Enter/Space: abort",
        "LOCK":   "Enter/Space: stop   r:record",
        "JOG":    "WASD  n:switch  Enter:exit",
    }.get(mode, "t:targets c:char l:lock j:jog  r:record  a:analyze  p:plot  h:help  q:quit")

    cv2.putText(scaled, keys, (sw//2 - 200, sh - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

    if show_help:
        help_lines = [
            "t  Target setting (WASD + mouse)",
            "c  Characterisation (auto-plots when done)",
            "l  Closed-loop lock",
            "j  Pixel-space jog",
            "r  Start/stop recording (open or closed)",
            "a  Plot OL vs CL time series + PSD",
            "p  Plot Jacobian / response curves",
            "h  Toggle help",
            "Space  Emergency stop",
            "q  Quit",
        ]
        cv2.rectangle(scaled, (8, 8), (430, 20 + 26*len(help_lines)), (15,15,15), -1)
        for i, line in enumerate(help_lines):
            cv2.putText(scaled, line, (18, 32 + i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 1)

    return scaled


def load_latest_control_matrix() -> Optional[np.ndarray]:
    files = sorted(glob.glob("actuator_characterization_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    print(f"Loaded control matrix from {files[-1]}")
    return np.array(data["control_matrix"])

# =====================================================================
# 9. Main
# =====================================================================

def main():
    cam_cfg = CameraConfig()
    det_cfg = DetectionConfig()
    char_cfg = CharacterisationConfig()

    stage = Newport.Picomotor8742()
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_roi(*cam_cfg.roi)
    cam.set_exposure(cam_cfg.exposure)
    cam.set_frame_period(cam_cfg.frame_period)
    print("Camera info:", cam.get_device_info())

    target_setter = TargetSetter(move_step=8)
    characteriser = Characteriser(char_cfg)

    M = load_latest_control_matrix()
    if M is not None:
        characteriser.control_matrix = M
        print("Previous control matrix ready")
    else:
        print("No previous matrix – run characterisation first")

    cl = ClosedLoop(characteriser)
    beam_jog = BeamJog(characteriser)
    recorder = DataRecorder()
    analysis_mode = False
    show_help = False
    display_scale = 0.6

    cam.start_acquisition()
    print("\nLive view started.")
    print("  t targets | c char | l lock | j jog")
    print("  r record  | a analyze OL/CL | p plot Jacobian")
    print("  h help    | Space emergency stop | q quit")

    cv2.namedWindow("Target Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Target Setup", 960, 720)

    # Trackbars for live detection tuning
    def on_threshold(val):
        det_cfg.threshold = max(1, val)
    def on_contrast(val):
        det_cfg.contrast_alpha = max(0.5, val / 10.0)

    cv2.createTrackbar("threshold", "Target Setup", det_cfg.threshold, 40, on_threshold)
    cv2.createTrackbar("contrast x10", "Target Setup", int(det_cfg.contrast_alpha * 10), 100, on_contrast)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and target_setter.active:
            target_setter.set_target_from_mouse(x / display_scale, y / display_scale)

    cv2.setMouseCallback("Target Setup", on_mouse)

    try:
        while True:
            cam.wait_for_frame(timeout=1.0)
            frames = cam.read_multiple_images()
            if not frames:
                continue
            frame = frames[-1]

            centroids, processed = process_frame(frame, det_cfg)
            h, w = processed.shape[:2]

            key = cv2.waitKey(1) & 0xFF
            commands: List[Command] = []

            # Global keys
            if key == ord('h'):
                show_help = not show_help
            if key == ord(' ') or key == 27:
                target_setter.stop()
                characteriser.stop()
                cl.stop()
                beam_jog.stop()
                if recorder.active:
                    recorder.stop()
                print(">>> EMERGENCY STOP")

            # Recording toggle
            if key == ord('r'):
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(is_closed_loop=cl.active)

            # Analysis & plots
            # --- Analysis mode toggle ---
            if key == ord('a'):
                if analysis_mode:
                    # Leave analysis mode
                    analysis_mode = False
                    plt.close('all')
                    print("Returned to live view")
                else:
                    # Enter analysis mode
                    if recorder.has_data():
                        analysis_mode = True
                        print("\n>>> ANALYSIS MODE")
                        recorder.plot_analysis(mode="both")   # uses data already in memory
                    else:
                        print("No recordings in memory. Press 'r' to record first.")

            # Also allow Esc to leave analysis mode
            if key == 27 and analysis_mode:       # Esc
                analysis_mode = False
                plt.close('all')
                print("Returned to live view")
            if key == ord('p'):
                characteriser.plot_responses()

            # Mode dispatch
            if key == ord('t') and not any([cl.active, characteriser.active, target_setter.active, beam_jog.active]):
                target_setter.start(centroids, (w, h))

            if target_setter.active:
                _, cmds = target_setter.handle(key, centroids)
                commands.extend(cmds)

            if key == ord('c') and not any([characteriser.active, target_setter.active, cl.active, beam_jog.active]):
                characteriser.start()

            if characteriser.active:
                _, cmds = characteriser.handle(centroids)
                commands.extend(cmds)

            # Errors
            errors: Dict[int, SpotError] = {}
            if centroids is not None and len(centroids) == 2:
                targets = target_setter.get_targets()
                for i in (0, 1):
                    if targets[i] is not None:
                        dx = centroids[i][0] - targets[i][0]
                        dy = centroids[i][1] - targets[i][1]
                        errors[i] = SpotError(dx=dx, dy=dy, dist=(dx**2 + dy**2)**0.5)

            if cl.active and len(errors) == 2:
                _, cmds = cl.handle(errors, centroids)
                commands.extend(cmds)

            if key == ord('l') and not any([target_setter.active, beam_jog.active, characteriser.active, cl.active]):
                cl.start()

            if key == ord('j') and not any([target_setter.active, characteriser.active, cl.active, beam_jog.active]):
                beam_jog.start()

            if beam_jog.active:
                _, cmds = beam_jog.handle(key)
                commands.extend(cmds)

            if not analysis_mode:
                # Hardware
                for cmd in commands:
                    if isinstance(cmd, MoveAxis):
                        stage.move_by(cmd.axis, cmd.steps)
                        stage.wait_move(cmd.axis)

            # Recorder update (every frame)
            recorder.update(centroids)

            # Display state
            if target_setter.active:
                mode, active_spot, last_steps = "TARGET", target_setter.current_spot, None
            elif characteriser.active:
                mode, active_spot, last_steps = "CHAR", None, None
            elif cl.active:
                mode, active_spot, last_steps = "LOCK", None, cl.last_steps
            elif beam_jog.active:
                mode, active_spot, last_steps = "JOG", beam_jog.current_spot, beam_jog.last_steps
            else:
                mode, active_spot, last_steps = "IDLE", None, None

            live = build_display(
                processed, centroids, target_setter.get_targets(),
                active_spot, det_cfg,
                mode=mode,
                errors=errors if errors else None,
                char_status=characteriser.status_text() if characteriser.active else "",
                last_steps=last_steps,
                show_help=show_help,
                recording=recorder.active,
                rec_label=recorder.label if recorder.active else "",
                scale=display_scale,
            )

            if analysis_mode:
                cv2.putText(live, "ANALYSIS MODE – press A or Esc to return",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Target Setup", live)
            if key == 13:
                target_setter.stop()
                characteriser.stop()
                cl.stop()
                beam_jog.stop()

            if key == ord('q'):
                break

    finally:
        cam.stop_acquisition()
        cv2.destroyAllWindows()
        plt.close("all")


if __name__ == "__main__":
    main()