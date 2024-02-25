import os
import numpy as np
import subprocess
import argparse
import pandas as pd
import datetime
import mne
from scipy.signal import find_peaks

from ExtractFrameTimestamps import ExtractTimestamps
from EstimateEyeCursor import EstimateCursor
from predict_deepsort import ObjectDetection, check_cuda_available
from query_yes_no import query_yes_no

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

def query_cmd(cmd_str, output_path, force_overwrite=False):
    if force_overwrite:                     _ = subprocess.check_output(cmd_str, shell=True)
    elif not os.path.exists(output_path):   _ = subprocess.check_output(cmd_str, shell=True)
    elif query_yes_no(f"The file \"{output_path}\" already exists. Do you wish to overwrite?", default=None):
                                            _ = subprocess.check_output(cmd_str, shell=True)
    else:                                   print(f"\tDid not generate a new \"{output_path}\"")

def ProcessFootage(root, camera, mappings, vr_events, eeg, camera_start_ms, objdet_model, force_overwrite=False, verbose=True):

     
    """ ================================================================
    === Step 1: Build relative URLs for each of the necessary files ====
    ==================================================================== """
    camera_path = os.path.join(root, camera)
    camera_ext = os.path.splitext(camera)[1]
    mappings_path = os.path.join(root, mappings)
    vr_events_path = os.path.join(root, vr_events)
    eeg_path = os.path.join(root, eeg)

    """ ===================================
    === Step 2: Trim the camera footage ===
    =======================================
    How to trim: we need the starting position (in hh:mm:ss.milli) and the end (in hh:mm:ss.milli)
    To calculate the start, we perform: simulation_start - camera_start_ms
    The choice in `simulation_start` is up to us. To avoid noise, let's trim so that
    we get all the data starting from the 2nd trial. The first trial started inside, after all. """
    events_df = pd.read_csv(vr_events_path)
    #start_timestamp = events_df.iloc[0]['unix_ms']
    #end_timestamp = events_df.iloc[-1]['unix_ms']
    trial_rows = events_df[events_df['title'].str.contains('Trial', regex=True, na=False)]
    start_timestamp = trial_rows.iloc[1]['unix_ms']
    end_timestamp = trial_rows.iloc[-2]['unix_ms']
    if end_timestamp <= start_timestamp:
        print("ERROR: The 2nd-to-last trial and the 2nd trial overlap or are the same. There's not enough data to handle. This user's data cannot be used.")
        return None
    trim_start_ms = (start_timestamp - camera_start_ms)/1000
    trim_end_ms = (end_timestamp - camera_start_ms)/1000
    trim_path = os.path.join(root, f'trim{camera_ext}')
    cmd_str = f'ffmpeg -i {camera_path} -vf "trim=start={trim_start_ms}:end={trim_end_ms},setpts=PTS-STARTPTS" -an -vsync 2 {trim_path}'
    query_cmd(cmd_str, trim_path, force_overwrite)

    """ ================================================
    === Step 3: Extract the timestamps and save them ===
    ================================================ """
    timestamps_path = ExtractTimestamps(trim_path, 0.0, True, verbose)

    """ ====================================================================
    === Step 4: Correct the footage, Get the left and right eye footage ====
    ========================================================================
    We'll get both left and right eyes for this one
    We'll be performing cropping, rotating, and lens correction via Hugin's method """
    left_path = os.path.join(root, f'left{camera_ext}')
    cmd_str = f'ffmpeg -i {trim_path} -vf "crop=632:672:16:0,rotate=21*(PI/180),lenscorrection=cx=0.57:cy=0.51:k1=-0.48:k2=0.2" -vsync 2 {left_path}'
    query_cmd(cmd_str, left_path, force_overwrite)
    left_eye_timestamps_path = ExtractTimestamps(left_path, 0.0, True, verbose)

    right_path = os.path.join(root, f'right{camera_ext}')
    cmd_str = f'ffmpeg -i {trim_path} -vf "crop=632:672:648:0,rotate=-21*(PI/180),lenscorrection=cx=0.43:cy=0.51:k1=-0.48:k2=0.2" -vsync 2 {right_path}'
    query_cmd(cmd_str, right_path, force_overwrite)
    right_eye_timestamps_path = ExtractTimestamps(left_path, 0.0, True, verbose)

    """ ============================
    === Step 5: object detection ===
    ================================
    we need to execute object detection on this.
    We might go with YOLOV5 as our object detection. 
    However, I want to also employ deepsort as a method to see if we can 
        stabilize the object tracking """
    gpu_available = check_cuda_available()
    deepsort_detector = ObjectDetection(objdet_model)
    objdet_path, objdet_vidpath = deepsort_detector(left_path, root, gpu_available, force_overwrite=force_overwrite)
    objdet_timestamps_path = ExtractTimestamps(objdet_vidpath, 0.0, True, verbose)

    """ ===========================================
    === Step 6: eye cursor estimation and depth ===
    ===============================================
    For each frame, we need to estimate what the position of the eye might be
    We'll use `EstimateEyeCusor` and only the left eye for this """
    cursor_vidpath, cursor_csvpath = EstimateCursor(
        objdet_vidpath, 
        vr_events_path, 
        mappings_path, 
        timestamps_path,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        force_overwrite=force_overwrite
            )
    cursor_timestamps_path = ExtractTimestamps(cursor_vidpath, 0.0, True, verbose)
    
    """ ==============================================================
    === Step 7: Limit the EEG data range based on video start time ===
    ============================================================== """
    start_timestamp_sec = start_timestamp/1000
    eeg_df = pd.read_csv(eeg_path)
    eeg_df.rename(columns={'ch1':'TP9', 'ch2':'AF7', 'ch3':'AF8', 'ch4':'TP10', 'ch5':'AUX'}, inplace=True)
    eeg_df = eeg_df.drop_duplicates()
    eeg_df = eeg_df[(eeg_df['unix_ts'] >= start_timestamp_sec) & (eeg_df['unix_ts'] <= (end_timestamp/1000))]
    eeg_df['unix_rel_ts'] = eeg_df['unix_ts'] - start_timestamp_sec
    eeg_df.to_csv(os.path.join(root,'eeg_trimmed.csv'), index=False)
    
    """ =============================================
    === Step 8: Process EEG data into mne package ===
    ============================================= """
    eeg_start = eeg_df.iloc[0]['unix_ts']
    eeg_end = eeg_df.iloc[-1]['unix_ts']
    eeg_duration = eeg_end - eeg_start
    eeg_size = len(eeg_df.index)
    eeg_frequency = round(eeg_size / eeg_duration)
    eeg_info = mne.create_info(["TP9","TP10","AF7", "AF8"], eeg_frequency, ch_types='eeg', verbose=False)
    s_array = np.transpose(eeg_df[["TP9", "TP10", "AF7", "AF8"]].to_numpy())
    mne_info = mne.io.RawArray(s_array, eeg_info, first_samp=0, copy='auto', verbose=False)
    mne_info.filter(0, 100, verbose=False)
    print('eeg_start: ' + str(eeg_start))
    print('eeg_end: ' + str(eeg_end))
    print('eeg_duration: ' + str(eeg_duration))

    """ ======================================
    === Step 9: Parsing EEG frame-by-frame ===
    ==========================================
    Here, we will attempt to parse the EEG at each known frame.
    The frame timings will be designated at each frame timestamp.
    The choice of sliding window size (in time) is a difficult choice.
    We'll go with a timestamp of a 2-sec timestamp. That should theoretically balance the temporal and spatial resolution of the PSD.
    We'll include samples from only AFTER the current timestamp. While having samples from both before and after the timestamp...
    ... might provide temporal context, we're more interested in the rapid changes in EEG signals. So the temporal context is not...
    ... going to be very useful. We'll still ahve the signal data on record for each frame anyway :shrug. """
    frequencies=["delta","theta","alpha","beta","gamma"]
    frequency_bands = {
        "delta": {"range":(0.5,4),"color":"darkgray"},
        "theta": {"range":(4, 8),"color":"lightblue"},
        "alpha": {"range":(8, 16),"color":"blue"},
        "beta":  {"range":(16, 32),"color":"orange"},
        "gamma": {"range":(32, 80),"color":"red"}
    }
    frame_timestamps = pd.read_csv(timestamps_path)
    frame_timestamps_list = frame_timestamps['timestamp'].to_list()
    for eeg_start in frame_timestamps_list:
        eeg_end = eeg_start + 2.0
        if eeg_end > frame_timestamps_list[-1]: break
        psd = mne_info.compute_psd(tmin=eeg_start, tmax=eeg_end, average='mean', verbose=False)
        powers, freqs = psd.get_data(picks=["AF7", "AF8"], return_freqs=True)
        peak_freqs = {}
        peak_powers = {}
        for freq in frequencies:
            peak_freqs[freq] = []
            peak_powers[freq] = []
        if len(powers) > 0:
            for i in range(len(["AF7", "AF8"])):
                peaks, _ = find_peaks(powers[i], threshold=50)
                for peak in peaks:
                    peak_freq = freqs[peak]
                    peak_power = powers[i][peak]
                    for freq in frequencies:
                        if peak_freq >= frequency_bands[freq]["range"][0] and peak_freq < frequency_bands[freq]["range"][1]:
                            peak_freqs[freq].append(peak_freq)
                            peak_powers[freq].append(peak_power)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="The root directory of the user data. Must be a directory.")
    parser.add_argument('camera', help="The filename of the raw camera footage. It must be the RAW footage, aka nothing has been croppped or anything.")
    parser.add_argument("mappings", help="The filename of the mappings for eye cursor correction.")
    parser.add_argument("vr_events", help="The filename of the csv file that stores the events captured from the VR simulation app.")
    parser.add_argument("eeg", help="The filename of the eeg data.")

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
        args.eeg,
        args.camera_start_ms,
        args.model,
        args.force,
        args.verbose
            )

