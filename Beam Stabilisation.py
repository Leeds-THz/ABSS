#!/usr/bin/env python3
"""
Beam stabilisation – pure motor-axis control (no Jacobian, no closed-loop)

- Target setting (individual / coupled + persistent snapshot history)
- All historical targets are drawn on screen
- Motor jog with couple/decouple
- Live inter-spot distance + error vectors
- Open-loop recording + time-series / PSD analysis
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union
import os
import json
import math
import numpy as np
import cv2
from scipy import ndimage, signal
import matplotlib.pyplot as plt

from pylablib.devices import Thorlabs, Newport
from collections import deque, defaultdict
import pandas as pd


# =====================================================================
# 1. Configuration
# =====================================================================

log = {
    "t": [],          # time since start (s)
    "motor": [],      # which motor (1-4)
    "steps": [],      # signed steps commanded
    "spot": [],       # which spot (0 or 1)
    "axis": [],       # 0 = x, 1 = y
    "px": [],         # current pixel x of the spot
    "py": [],         # current pixel y of the spot
    "error": [],      # signed error on the controlled axis
}

@dataclass(frozen=True)
class CameraConfig:
    roi: Tuple[int, int, int, int] = (0, 1440, 0, 1080)
    exposure: float = 0.02
    frame_period: float = 0.035

from dataclasses import dataclass

@dataclass
class SpotError:
    dx: float
    dy: float

    @property
    def dist(self) -> float:
        return float(np.hypot(self.dx, self.dy))
    
@dataclass
class DetectionConfig:
    min_spot_area: int = 50
    threshold: int = 3
    gamma: float = 1.1
    contrast_alpha: float = 4.0
    contrast_beta: float = 20.0

# =====================================================================
# 2. Narrow command interface
# =====================================================================

Point = Tuple[float, float]

@dataclass
class Finished:
    reason: str = ""

@dataclass
class MoveAxis:
    axis: int
    steps: int

Command = Union[MoveAxis, Finished]


# =====================================================================
# 3. Pure vision pipeline
# =====================================================================

def preprocess_image(gray: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    # Subtract mean background, apply gamma correction
    background = np.mean(gray)
    gray = gray.astype(np.float32) - background
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = (gray / 255.0) ** cfg.gamma
    gray = (gray * 255).astype(np.uint8)
    return gray

def detect_two_spots(gray: np.ndarray, cfg: DetectionConfig) -> List[Tuple]:
    # Thresold connected components. Keeping the two largest spots.
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
    # Intensity-weighted centre-of-mass inside bounding box of the spot
    # define region of interest for beam spot
    x, y, w, h = roi
    roi_img = gray[y:y+h, x:x+w].astype(np.float32)
    roi_img[roi_img < cfg.threshold] = 0
    if np.sum(roi_img) == 0:
        return None
    cy, cx = ndimage.center_of_mass(roi_img)
    return (cx + x, cy + y)


def process_frame(frame: np.ndarray, cfg: DetectionConfig) -> Tuple[Optional[List[Point]], np.ndarray]:
    # Full pipeline: 16-bit -> 8-bit conversion -> contrast stretch -> preprocess -> detect -> centroids
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(frame).squeeze()

    # 12-bit value turned to 8-bit value
    if gray.dtype == np.uint16:
        gray = (gray >> 8).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    gray = cv2.convertScaleAbs(gray, alpha=cfg.contrast_alpha, beta=cfg.contrast_beta)
    processed = preprocess_image(gray, cfg)

    spots = detect_two_spots(processed, cfg)
    # if len(spots) != 2:
    #     return None, processed

    centroids = []
    for spot in spots:
        x, y, w, h, _ = spot
        c = calculate_centroid(processed, (x, y, w, h), cfg)
        if c is not None:
            centroids.append(c)

    # if len(centroids) != 2:
    #     return None, processed
    return centroids, processed

# =====================================================================
# Beam Correction
# =====================================================================

def GetCurrentCameraFrame(cam):
    # Block until at least one new frame is available and returns the 
    # Loop until there are some frames to get
    while(True):
        frames = cam.read_multiple_images()
        if frames != None and len(frames) > 0:
            break
    return frames[-1]

def GetCurrentSpot(cam, spotNo):
    # Returns [x, y] of the requested spot (0 or 1)
    # Assumes the two largest spots are ordered consistently
    
    det_cfg = DetectionConfig()
    curFrame = GetCurrentCameraFrame(cam)
    centroids, _ = process_frame(curFrame, det_cfg)
    SpotAmount = len(centroids)
    if SpotAmount == 2:
        # Acquire the desired spot
        x1, y1 = centroids[0]
        x2, y2 = centroids[1]
        # Return format [x, y]
        return [centroids[spotNo][0], centroids[spotNo][1]]
    elif SpotAmount == 1:
        return centroids[0][0], centroids[0][1]

    # elif len(centroids) == 0:
    #     return None
    # TODO: Add error handling for no spots

def GetSpotAmount(cam):
    det_cfg = DetectionConfig()
    curFrame = GetCurrentCameraFrame(cam)
    centroids, _ = process_frame(curFrame, det_cfg)
    SpotAmount = len(centroids)
    return SpotAmount

def GetSpotError(cam, targetDict, spotNo, axis):
    # Load the target position / image
    curTarget = targetDict["Spot" + str(spotNo)]
    
    # Acquire the current camera image
    curSpot = GetCurrentSpot(cam, spotNo)
    
    # Get the error (manhattan distance)
    # curError = abs(curTarget[0] - curSpot[0]) + abs(curTarget[1] - curSpot[1])
    curError = curSpot[axis] - curTarget[axis]

    return curError, curSpot

def SetTargetPositionsFromCurrentFrame(cam):
    # Get current camera frame
    # GetCurrentCameraFrame()

    # Acquire the current camera frame
    det_cfg = DetectionConfig()
    curFrame = GetCurrentCameraFrame(cam)
    centroids, _ = process_frame(curFrame, det_cfg)

    if len(centroids) == 1:
        return {"Spot0": [centroids[0][0], centroids[0][1]], "Spot1": [0, 0]}

    # Return centroids of both spots as a dict
    return {"Spot0": [centroids[0][0], centroids[0][1]], "Spot1": [centroids[1][0], centroids[1][1]]}

def DebugDisplay(cam, targetDict):
    # Display the current camera image with the target positions overlayed
    a = 1
    det_cfg = DetectionConfig()
    # curFrame = GetCurrentCameraFrame(cam)
    cv2.namedWindow("Live Beam Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Beam Display", 960, 720)
    frame = GetCurrentCameraFrame(cam)
    centroids, processed = process_frame(frame, det_cfg)
    display = cv2.convertScaleAbs(processed, 4.0, 20.0)
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    for i, key in enumerate(["Spot0", "Spot1"]):
        # Live spot (green)
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)

        # Target (cyan)
        tx, ty = int(targetDict[key][0]), int(targetDict[key][1])
        cv2.circle(display, (tx, ty), 10, (0, 255, 180), -1)

        # Error line + distance text
        dist = np.hypot(centroids[i][0] - targetDict[key][0],
                        centroids[i][1] - targetDict[key][1])
        colour = (0, 255, 0) if dist < 5 else (0, 200, 255) if dist < 20 else (0, 80, 255)
        cv2.line(display, (tx, ty), (cx, cy), colour, 2)
        cv2.putText(display, f"{dist:.1f}px",
                    (tx + 12, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    cv2.imshow("Live Beam Display", display)

def BeamAlignment(stage, cam, swapSpots=False,
                  errThresh=2.0, max_steps=50, singlePass=True):
    """
    Pure motor-axis proportional control.

    singlePass=True  → one clean pass, then stop when settled
    singlePass=False → continuous correction until lost spots or Ctrl-C
    """
    targetDict = SetTargetPositionsFromCurrentFrame(cam)

    motorAxisLUT = [1, 0, 1, 0]      # motor → axis (0=x, 1=y)
    KpLUT = [5.0, 5.0, 5.0, 5.0]

    log = {
        "t": [], "motor": [], "steps": [], "spot": [],
        "axis": [], "px": [], "py": [], "error": []
    }
    t0 = time.time()

    def log_move(i, curSpotNo, steps_cmd, prevError):
        cur_spot = GetCurrentSpot(cam, curSpotNo)
        log["t"].append(time.time() - t0)
        log["motor"].append(i + 1)
        log["steps"].append(steps_cmd)
        log["spot"].append(curSpotNo)
        log["axis"].append(motorAxisLUT[i])
        log["px"].append(cur_spot[0])
        log["py"].append(cur_spot[1])
        log["error"].append(prevError)

    try:
        while True:
            

            motors_moved = False

            for i in range(4):
                curSpotNo = 0 if i < 2 else 1
                if swapSpots:
                    curSpotNo = 1 - curSpotNo

                if GetSpotAmount(cam) < 2:
                    if singlePass:
                        print("Lost spots – aborting")
                        return -3
                    else:
                        break

                prevError, _ = GetSpotError(cam, targetDict, curSpotNo, motorAxisLUT[i])

                if abs(prevError) > errThresh:
                    steps_cmd = int(round(KpLUT[i] * prevError))

                    if steps_cmd != 0:
                        stage.move_by(i + 1, steps_cmd)
                        stage.wait_move()
                        motors_moved = True
                        # log_move(i, curSpotNo, steps_cmd, prevError)

            if not motors_moved:
                if singlePass:
                    break                   # finished one clean pass
                time.sleep(0.05)
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl-C)")
    finally:
        pass
    #     if log["t"]:
    #         pd.DataFrame(log).to_csv("motor_pixel_log.csv", index=False)
    #         print(f"Logged {len(log['t'])} moves → motor_pixel_log.csv")
    #     stage.stop()

    return 0

def plot_pixel_vs_time(csv_path="motor_pixel_log.csv"):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for spot in [0, 1]:
        mask = df["spot"] == spot
        axes[0].plot(df.loc[mask, "t"], df.loc[mask, "px"], '.-', label=f"Spot{spot} X")
        axes[1].plot(df.loc[mask, "t"], df.loc[mask, "py"], '.-', label=f"Spot{spot} Y")

    axes[0].set_ylabel("Pixel X")
    axes[1].set_ylabel("Pixel Y")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title("Spot position vs time")
    plt.tight_layout()
    plt.show()

def SettingsDictToDetectorConfig(settingsDict):
    return DetectionConfig(settingsDict["DetectorConfig"]["min_spot_area"])

def plot_motor_activity(csv_path="motor_pixel_log.csv"):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in sorted(df["motor"].unique()):
        mask = df["motor"] == m
        ax.stem(df.loc[mask, "t"], df.loc[mask, "steps"],
                        linefmt=f"C{m-1}-", markerfmt=f"C{m-1}o",
                        basefmt=" ", label=f"Motor {m}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Steps commanded")
        ax.set_title("Motor movements over time")
        ax.legend()
        ax.axhline(0, color='k', lw=0.5)
        plt.tight_layout()
        plt.show()
    # log motor movements in a csv file
    # log how laser spot moves during the day
if __name__ == "__main__":
    # main()
    cam = Thorlabs.ThorlabsTLCamera()
    cam_cfg = CameraConfig()
    stage = Newport.Picomotor8742()
    cam.set_roi(*cam_cfg.roi)
    cam.set_exposure(cam_cfg.exposure)
    cam.set_frame_period(cam_cfg.frame_period)
    cam.start_acquisition()

    # time.sleep(1)
    print("ABSS Starting...")
    # Single Pass Instantiatiom
    # Continuous stabilisation (what you want most of the time)
    BeamAlignment(stage, cam, swapSpots=True, errThresh=0.25, singlePass=False)
    # One-shot alignment
    # BeamAlignment(stage, cam, swapSpots=True, errThresh=0.5, singlePass=True)
    # plot_pixel_vs_time()
    # plot_motor_activity()