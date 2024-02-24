import os
import subprocess
import argparse
import pandas as pd
import datetime

from ExtractFrameTimestamps import ExtractTimestamps
from EstimateEyeCursor import EstimateCursor
from predict_deepsort import ObjectDetection, check_cuda_available

def ms_to_hours(millis, include_millis=True):
    seconds, milliseconds = divmod(millis, 1000)
    if len(str(seconds)) == 1:  seconds = f'0{seconds}'
    if len(str(milliseconds)) == 1: milliseconds = f'00{milliseconds}'
    elif len(str(milliseconds)) == 2: milliseconds = f'0{milliseconds}'
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if len(str(minutes)) == 1: minutes = f'0{minutes}'
    if len(str(hours)) == 1: hours = f'0{hours}'
    if include_millis:
        return f"{hours}:{minutes}:{seconds}.{milliseconds}"
    return f"{hours}:{minutes}:{seconds}"


def ProcessFootage(root, camera, mappings, vr_events, camera_start_ms, objdet_model, force_overwrite=False, verbose=True):

    # Step 1: Build relative URLs for each of the necessary files
    camera_path = os.path.join(root, camera)
    camera_ext = os.path.splitext(camera)[1]
    mappings_path = os.path.join(root, mappings)
    vr_events_path = os.path.join(root, vr_events)

    # Step 2: Trip the camera footage based on some calculations involving the start time of the video and the start time of the simulation
    # How to trim: we need the starting position (in hh:mm:ss.milli) and the end (in hh:mm:ss.milli)
    # To calculate the start, we perform: simulation_start - camera_start_ms
    # We get `simulation_start` from `vr_events.csv` - the first row value
    events_df = pd.read_csv(vr_events_path)
    simulation_start = events_df.iloc[0]['unix_ms']
    simulation_end = events_df.iloc[-1]['unix_ms']
    trim_start_ms = (simulation_start - camera_start_ms)/1000
    trim_end_ms = (simulation_end - camera_start_ms)/1000
    print(trim_start_ms, ' - ', trim_end_ms)
    trim_path = os.path.join(root, f'trim{camera_ext}')
    #cmd_str = f'ffmpeg -i {camera_path} -vf "trim=start={trim_start_ms}:end={trim_end_ms},setpts=PTS-STARTPTS" -af "atrim=start={trim_start_ms}:end={trim_end_ms},setpts=PTS-STARTPTS" -vsync 2 {trim_path}'
    if not os.path.exists(trim_path) or force_overwrite:
        cmd_str = f'ffmpeg -i {camera_path} -vf "trim=start={trim_start_ms}:end={trim_end_ms},setpts=PTS-STARTPTS" -an -vsync 2 {trim_path}'
        _ = subprocess.check_output(cmd_str, shell=True)

    # Step 3: Extract the timestamps and save them
    timestamps_path = ExtractTimestamps(trim_path, 0.0, True, verbose)

    # Step 4: Correct the footage
    # We'll get both left and right eyes for this one
    # We'll be performing cropping, rotating, and lens correction via Hugin's method
    left_path = os.path.join(root, f'left{camera_ext}')
    if not os.path.exists(left_path) or force_overwrite:
        cmd_str = f'ffmpeg -i {trim_path} -vf "crop=632:672:16:0,rotate=21*(PI/180),lenscorrection=cx=0.57:cy=0.51:k1=-0.48:k2=0.2" -vsync 2 {left_path}'
        _ = subprocess.check_output(cmd_str, shell=True)
    left_eye_timestamps_path = ExtractTimestamps(left_path, 0.0, True, verbose)
    right_path = os.path.join(root, f'right{camera_ext}')
    if not os.path.exists(right_path) or force_overwrite:
        cmd_str = f'ffmpeg -i {trim_path} -vf "crop=632:672:648:0,rotate=-21*(PI/180),lenscorrection=cx=0.43:cy=0.51:k1=-0.48:k2=0.2" -vsync 2 {right_path}'
        _ = subprocess.check_output(cmd_str, shell=True)
    right_eye_timestamps_path = ExtractTimestamps(left_path, 0.0, True, verbose)

    # Step 5: eye cursor estimation and depth
    # For each frame, we need to estimate what the position of the eye might be
    # We'll use `EstimateEyeCusor` and only the left eye for this
    cursor_vidpath, cursor_csvpath = EstimateCursor(left_path, vr_events_path, mappings_path, timestamps_path)
    cursor_timestamps_path = ExtractTimestamps(cursor_vidpath, 0.0, True, verbose)

    # Sttep 6: object detection
    # we need to execute object detection on this.
    # We might go with YOLOV5 as our object detection. However, I want to also employ deepsort as a method to see if we can stabilize the object tracking
    gpu_available = check_cuda_available()
    deepsort_detector = ObjectDetection(objdet_model)
    objdet_path, objdet_vidpath = deepsort_detector(left_path, os.path.join(root,'deepsort_results'), gpu_available)
    objdet_timestamps_path = ExtractTimestamps(objdet_vidpath, 0.0, True, verbose)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="The root directory of the user data. Must be a directory.")
    parser.add_argument('camera', help="The filename of the raw camera footage. It must be the RAW footage, aka nothing has been croppped or anything.")
    parser.add_argument("mappings", help="The filename of the mappings for eye cursor correction.")
    parser.add_argument("vr_events", help="The filename of the csv file that stores the events captured from the VR simulation app.")
    
    parser.add_argument("camera_start_ms", help="The start time (unix milliseconds) of when the camera feed was recorded. AKA at what unix millisecond was the camera footage started at?", type=int)
    parser.add_argument("model", help="What model should we use for the YOLO implmenetation and object detection/tracking?")
    parser.add_argument('-f', '--force', help="Force the creation of videos regardless if they exist or not", action="store_true")
    parser.add_argument('-v', '--verbose', help="Should we be verbose in printing out statements?", action="store_true")
    args = parser.parse_args()

    print(args)

    ProcessFootage(
        args.root, 
        args.camera, 
        args.mappings, 
        args.vr_events, 
        args.camera_start_ms,
        args.model,
        args.force,
        args.verbose
            )

