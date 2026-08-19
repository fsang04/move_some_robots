"""Moves the robot arm in a figure eight pattern and captures images of the AprilTag (right arm calibration)."""

import argparse
from pathlib import Path

import cv2
import k4a
import numpy as np
import pyzed.sl as sl
from crisp_py.robot import Robot
from scipy.spatial.transform import Rotation
import time


parser = argparse.ArgumentParser(description="Capture poses and images for calibration with specified camera (right arm).")
parser.add_argument(
    "--camera",
    type=str,
    choices=["azure", "zed"],
    default="azure",
    help="Camera to use: 'azure' or 'zed' (default: azure)",
)
parser.add_argument(
    "--seq-name",
    type=str,
    required=True,
    help="Calibration sequence name (data saved under captured_calibration_data/{seq_name}).",
)
args = parser.parse_args()
camera = args.camera
seq_name = args.seq_name

# initialize robot
left_arm = Robot(namespace="")
left_arm.wait_until_ready()

print("Going to home position...")
left_arm.home()

# figure eight parameters
radius = 0.2  # [m]
ctrl_freq = 50.0
sin_freq_y = 0.25  # rot / s
sin_freq_z = 0.125  # rot / s
max_time = 8.0

# Setup Params
initial_rotation = np.pi/2 # rotation arounnd robot z axis (counterclockwise)

left_arm.controller_switcher_client.switch_controller("cartesian_impedance_controller")
left_arm.cartesian_controller_parameters_client.load_param_config(
    file_path="/home/roahmlab/move_some_robots/crisp_env/crisp_py/config/control/default_cartesian_impedance.yaml"
)

# set initial target pose and orientation
print("Starting to draw a circle...")
t = 0.0
target_pose = left_arm.end_effector_pose.copy()
print("taget_pose_rotation:", target_pose.orientation)
original_rotation = np.array([  [1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0]])
turn_rotation = np.array([  [np.cos(initial_rotation), -1 * np.sin(initial_rotation), 0.0],
                                [np.sin(initial_rotation), np.cos(initial_rotation), 0.0],
                                [0.0, 0.0, 1.0]])
target_pose.orientation = Rotation.from_matrix(turn_rotation @ original_rotation)

center = np.array([0.3, 0.3, 0.4])
target_pose.position = center
print("taget_pose_rotation:", target_pose)
rate = left_arm.node.create_rate(ctrl_freq)
left_arm.move_to(pose=target_pose, speed=0.15)

# Setup camera. BOTH cameras now record depth: ZED via zed_calib_rgbd (MEASURE.DEPTH is
# already registered to VIEW.LEFT), Azure via k4a.Transformation.depth_image_to_color_camera.
if camera == "zed":
    azure_transformation = None
    # Depth-enabled ZED capture. The old path used RESOLUTION.AUTO (-> HD720), left
    # self-calibration on, saved 4-channel BGRA PNGs, and recorded NO depth, which is
    # why --use-depth-translation was blocked for the ZED. See zed_calib_rgbd.py.
    import zed_calib_rgbd
    zed, runtime_params, zed_info = zed_calib_rgbd.open_zed()
    image = None
else:
    print("Using Azure camera")
    device = k4a.Device.open()
    if device is None:
        exit(-1)
    device_config = k4a.DeviceConfiguration(
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_720P,
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
        camera_fps=k4a.EFramesPerSecond.FPS_30,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False)
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        exit(-1)
    cal = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    azure_transformation = k4a.Transformation.create(cal)


# data capture variables and output layout (same pattern as single_arm_capture: frame_list of {color, depth})
frame_count = 0
pose_count = 0
pose_list = []
frame_list = []  # each item: {"color": bgr_array, "depth": color-aligned depth or None}
DATAPATH = "/home/roahmlab/move_some_robots/crisp_env/crisp_py/hand_to_eye_calibration/roahm-deformable-objects"
base_dir = Path(DATAPATH) / "captured_calibration_data" / seq_name
frames_dir = base_dir / "frames"
base_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

# main trajectory loop
while t < max_time:
    
    if frame_count % 7 == 0:
        # Wait for arm to settle
        time.sleep(2.0)
        # Save the pose
        p = left_arm.end_effector_pose.copy()
        pose_list.append(np.array([p.position[0], p.position[1], p.position[2],
            p.orientation.as_quat()[0], p.orientation.as_quat()[1],
            p.orientation.as_quat()[2], p.orientation.as_quat()[3]]))
        
        # Take the image and save it (right arm); buffer color + color-aligned depth like single_arm_capture
        if camera == "zed":
            try:
                color_bgr, depth_data = zed_calib_rgbd.grab_rgbd(zed, runtime_params)
                cv2.imwrite(str(frames_dir / f"calibration_right_image_{pose_count}.png"), color_bgr)
                frame_list.append({"color": color_bgr, "depth": depth_data})
                print(f"Right image Captured {pose_count}")
                pose_count += 1
            except RuntimeError as exc:
                # Drop the pose just appended and do NOT advance pose_count, so image
                # index, pose_list index and depth_stack index stay aligned. The old
                # code incremented on failure while frame_list did not grow, which
                # offset every later frame's depth against its pose.
                print(f"ERROR: Failed to capture image {pose_count}: {exc} -- pose dropped")
                pose_list.pop()
        else:
            capture = device.get_capture(-1)
            color_image = capture.color
            color_image_data = color_image.data  # NumPy array (BGRA)
            color_bgr = cv2.cvtColor(color_image_data, cv2.COLOR_BGRA2BGR).copy()
            if capture.depth is not None and azure_transformation is not None:
                depth_color_img = azure_transformation.depth_image_to_color_camera(capture.depth)
                depth_data = depth_color_img.data.copy()  # (H, W) uint16, same H,W as color
            else:
                depth_data = None
            cv2.imwrite(str(frames_dir / f"calibration_right_image_{pose_count}.png"), color_bgr)
            frame_list.append({"color": color_bgr, "depth": depth_data})
            print(f"Right image Captured {pose_count}")
            pose_count += 1
            if status != k4a.EStatus.SUCCEEDED:
                exit(-1)


    frame_count += 1

    z_jitter = np.random.uniform(-0.05, 0.05)
    # z_jitter = 0
    
    # compute figure-eight trajectory position
    x = radius * np.sin(2 * np.pi * sin_freq_y * t) + center[0]
    y = center[1] + z_jitter
    z = radius * np.sin(2 * np.pi * sin_freq_z * t) + center[2] 
    target_pose.position = np.array([x, y, z])

    # send target to controller
    left_arm.set_target(pose=target_pose)
    rate.sleep()

    t += 1.0 / ctrl_freq

# save all poses for this right-arm calibration sequence
np.savez(base_dir / "right_calibration_poses.npz", *pose_list)

# save RGB-D in same format as single_arm_capture: rgbd.npz with color (N,H,W,3), depth (N,H,W) if Azure
n_saved = len(frame_list)
colors = np.stack([frame_list[i]["color"] for i in range(n_saved)], axis=0)  # (N,H,W,3)
# Camera-agnostic: store depth whenever the camera produced any. This used to be gated
# on `camera == "azure"`, so ZED captures silently dropped their depth even if present.
depth_list = [frame_list[i].get("depth") for i in range(n_saved)]
first_valid = next((d for d in depth_list if d is not None), None)
if first_valid is not None:
    H_d, W_d = first_valid.shape
    depth_stack = np.zeros((n_saved, H_d, W_d), dtype=first_valid.dtype)
    for i, d in enumerate(depth_list):
        if d is not None:
            depth_stack[i] = d
    # np.savez (uncompressed) on purpose: some tools memory-map 'depth' out of this npz.
    # disparity_offset_px = 0.0 records that this depth is RAW: zed_calib_rgbd.py
    # applies no correction. The solvers read this key and correct the depth
    # themselves. It is the guard against correcting twice, which would give 32 px
    # and a silently wrong calibration. If a capture path ever starts correcting at
    # capture time, it MUST write the offset it applied here instead of 0.0.
    np.savez(base_dir / "right_calibration_rgbd.npz", color=colors, depth=depth_stack,
             disparity_offset_px=np.float64(0.0))
    print(f"Saved right_calibration_rgbd.npz to {base_dir} with depth "
          f"{depth_stack.shape} {depth_stack.dtype} (0 = invalid)")
    # One viewable depth picture per pose, beside the colour frames. The values are
    # in the npz above; these are only so you can LOOK at the depth quality.
    import zed_calib_rgbd
    for i, d in enumerate(depth_list):
        if d is not None:
            zed_calib_rgbd.save_depth_vis(
                frames_dir / f"depth_right_image_{i}.png", d)
    print(f"Saved {sum(d is not None for d in depth_list)} depth pictures to {frames_dir}")
else:
    np.savez(base_dir / "right_calibration_rgbd.npz", color=colors)
    print(f"Saved right_calibration_rgbd.npz to {base_dir} (color only -- no depth)")

print("Waiting for robot to settle...")
time.sleep(1.0)
print("Done drawing a circle!")


print("return to home and shutdown")
left_arm.home()
left_arm.shutdown()
