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

import usb.core
import usb.backend.libusb1
import libusb_package

from dataclasses import dataclass

# =====================================================================
# Camera Image Processing 
# =====================================================================

def InitCamera(settingsDict):
    cam = Thorlabs.ThorlabsTLCamera()
    cam_cfg = settingsDict["CameraConfig"]
    cam.set_roi(*cam_cfg["roi"])
    cam.set_exposure(cam_cfg["exposure"])
    cam.set_frame_period(cam_cfg["frame_period"])
    cam.start_acquisition()

    return cam


def preprocess_image(gray: np.ndarray, cfg) -> np.ndarray:
    # Subtract mean background, apply gamma correction
    background = np.mean(gray)
    gray = gray.astype(np.float32) - background
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = (gray / 255.0) ** cfg["gamma"]
    gray = (gray * 255).astype(np.uint8)
    return gray

def detect_two_spots(gray: np.ndarray, cfg) -> List[Tuple]:
    # Thresold connected components. Keeping the two largest spots.
    _, binary = cv2.threshold(gray, cfg["threshold"], 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    spots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > cfg["min_spot_area"]:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            spots.append((x, y, w, h, area))

    spots = sorted(spots, key=lambda s: s[4], reverse=True)[:2]
    return spots


def calculate_centroid(gray: np.ndarray, roi: Tuple, cfg) -> Optional[Point]:
    # Intensity-weighted centre-of-mass inside bounding box of the spot
    # define region of interest for beam spot
    x, y, w, h = roi
    roi_img = gray[y:y+h, x:x+w].astype(np.float32)
    roi_img[roi_img < cfg["threshold"]] = 0
    if np.sum(roi_img) == 0:
        return None
    cy, cx = ndimage.center_of_mass(roi_img)
    return (cx + x, cy + y)


def process_frame(frame: np.ndarray, cfg) -> Tuple[Optional[List[Point]], np.ndarray]:
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

    gray = cv2.convertScaleAbs(gray, alpha=cfg["contrast_alpha"], beta=cfg["contrast_beta"])
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

def GetCurrentCameraFrame(cam):
    # Block until at least one new frame is available and returns the 
    # Loop until there are some frames to get
    while(True):
        frames = cam.read_multiple_images()
        if frames != None and len(frames) > 0:
            break
    return frames[-1]

def GetCurrentSpot(cam, settingsDict, spotNo):
    # Returns [x, y] of the requested spot (0 or 1)
    # Assumes the two largest spots are ordered consistently
    
    det_cfg = settingsDict["DetectorConfig"]
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

def GetSpotAmount(cam, settingsDict):
    det_cfg = settingsDict["DetectorConfig"]
    curFrame = GetCurrentCameraFrame(cam)
    centroids, _ = process_frame(curFrame, det_cfg)
    SpotAmount = len(centroids)
    return SpotAmount

def GetSpotError(cam, settingsDict, spotNo, axis):
    # Load the target position / image
    curTarget = settingsDict["Targets"]["Spot" + str(spotNo)]
    
    # Acquire the current camera image
    curSpot = GetCurrentSpot(cam, settingsDict, spotNo)
    
    # Get the error (manhattan distance)
    # curError = abs(curTarget[0] - curSpot[0]) + abs(curTarget[1] - curSpot[1])
    curError = curSpot[axis] - curTarget[axis]

    return curError, curSpot

# =====================================================================
# Beam Correction
# =====================================================================

def SetTargetPositionsFromCurrentFrame(settingsDict, cam):
    # Get current camera frame
    # GetCurrentCameraFrame()

    # Acquire the current camera frame
    # det_cfg = DetectionConfig()
    det_cfg = settingsDict["DetectorConfig"]
    curFrame = GetCurrentCameraFrame(cam)
    centroids, _ = process_frame(curFrame, det_cfg)

    if len(centroids) == 1:
        # return {"Spot0": [centroids[0][0], centroids[0][1]], "Spot1": [0, 0]}
        settingsDict["Targets"]["Spot0"] = [centroids[0][0], centroids[0][1]]
        settingsDict["Targets"]["Spot1"] = [0, 0]
    elif len(centroids) == 2:
        settingsDict["Targets"]["Spot0"] = [centroids[0][0], centroids[0][1]]
        settingsDict["Targets"]["Spot1"] = [centroids[1][0], centroids[1][1]]

    # Return centroids of both spots as a dict contained in settingsDict
    return settingsDict

def BeamAlignment(stage, cam, settingsDict, swapSpots=False,
                  errThresh=2.0, max_steps=50, singlePass=True):
    """
    Pure motor-axis proportional control.

    singlePass=True  → one clean pass, then stop when settled
    singlePass=False → continuous correction until lost spots or Ctrl-C
    """

    log = {
        "t": [], "motor": [], "steps": [], "spot": [],
        "axis": [], "px": [], "py": [], "error": []
    }
    t0 = time.time()

    # def log_move(i, curSpotNo, steps_cmd, prevError):
    #     cur_spot = GetCurrentSpot(cam, curSpotNo)
    #     log["t"].append(time.time() - t0)
    #     log["motor"].append(i + 1)
    #     log["steps"].append(steps_cmd)
    #     log["spot"].append(curSpotNo)
    #     log["axis"].append(motorAxisLUT[i])
    #     log["px"].append(cur_spot[0])
    #     log["py"].append(cur_spot[1])
    #     log["error"].append(prevError)

    try:
        while True:
            
            motors_moved = False

            # Iterate through the motor axes
            for i in range(4):
                # Set which spot it being checked for this motor
                curSpotNo = 0 if i < 2 else 1
                if swapSpots:
                    curSpotNo = 1 - curSpotNo


                while True:
                    # Check if there are two spots on the camera
                    # If there isn't - do nothing
                    if GetSpotAmount(cam, settingsDict) < 2:
                        if singlePass:
                            print("Lost spots – aborting")
                            return -3
                        else:
                            break

                    # Get the current error for the given motor axis
                    prevError, _ = GetSpotError(cam, settingsDict, curSpotNo, settingsDict["ControllerSettings"]["MotorAxis"][i])

                    # If the error is above threshold
                    if abs(prevError) > errThresh:
                        # Move motor to correct for error
                        steps_cmd = int(round(settingsDict["ControllerSettings"]["Kp"][i] * prevError))

                        if steps_cmd != 0:
                            stage.move_by(i + 1, steps_cmd)
                            stage.wait_move()
                            motors_moved = True
                            # log_move(i, curSpotNo, steps_cmd, prevError)
                    # If there is no more error in this axis, move onto the next axis
                    else:
                        break

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

# =====================================================================
# Init
# =====================================================================

def Init():
    settingsDict = LoadSettings()
    
    # Acquire the libusb.dll (from libusb_package) and set it as the usb backend
    libusb1_backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    usb_devices = usb.core.find(backend=libusb1_backend, find_all=True)
    
    stage = Newport.Picomotor8742()
    
    cam = InitCamera(settingsDict)

    return settingsDict, stage, cam

# =====================================================================
# Settings
# =====================================================================

def LoadSettings(settingsFile: String = "settings.json"):
    with open(settingsFile, 'r') as f:
        return json.load(f)

def SaveSettings(settingsDict, settingsFile: String = "settings.json"):
    jsonStr = json.dumps(settingsDict, indent=4)
    with open(settingsFile, 'w') as f:
            return f.write(jsonStr) 

# =====================================================================
# Tests
# =====================================================================


def SinglePassTest():
    # Init
    settingsDict, stage, cam = Init()

    # Set current position as target
    settingsDict = SetTargetPositionsFromCurrentFrame(settingsDict, cam)

    # Take snapshot
    startFrame = GetCurrentCameraFrame(cam)

    plt.imshow(startFrame)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    # cv2.imwrite('start.png', startFrame)

    # Wait for the user to misalign system
    print("Press Enter after system is misaligned...")
    input()

    # Single bass alignment
    BeamAlignment(stage, cam, settingsDict, swapSpots=settingsDict["ControllerSettings"]["SwapSpots"], errThresh=settingsDict["ControllerSettings"]["ErrorThreshold"], singlePass=True)

    # Take snapshot
    endFrame = GetCurrentCameraFrame(cam)
    # cv2.imwrite('end.png', endFrame)
    plt.imshow(endFrame)
    plt.axis('off')  # Turn off axis labels
    plt.show()

def LoopTest():
    settingsDict, stage, cam = Init()
    
    settingsDict = SetTargetPositionsFromCurrentFrame(settingsDict, cam)
    
    # time.sleep(1)
    print("ABSS Starting...")
    # Single Pass Instantiatiom
    # Continuous stabilisation (what you want most of the time)
    BeamAlignment(stage, cam, settingsDict, swapSpots=settingsDict["ControllerSettings"]["SwapSpots"], errThresh=settingsDict["ControllerSettings"]["ErrorThreshold"], singlePass=False)
    # One-shot alignment
    # BeamAlignment(stage, cam, swapSpots=True, errThresh=0.5, singlePass=True)
    # plot_pixel_vs_time()
    # plot_motor_activity()
    SaveSettings(settingsDict)

# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    SinglePassTest()
