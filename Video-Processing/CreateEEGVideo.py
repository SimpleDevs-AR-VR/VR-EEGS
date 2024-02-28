import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import mne

# These three functions sort filenames "humanly" (as opposed to the default lexicographical sorting that computers understand)
# The user-friendly function to use is `sort_nicely(l)`, where `l` is a list of files where all contents are of type ____<#>.png
# Source: https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks. "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def sort_nicely(l):
   return sorted(l, key=alphanum_key) 


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def CreateEEGVideo(eeg_path, output_dir, start_timestamp, end_timestamp, timestamps_path, fps=60, verbose=True):

    output_trimmed_csv = os.path.join(output_dir,'eeg_trimmed.csv')
    psd_videopath = os.path.join(root, 'psd.avi')

    """ ==============================================================
    === Step 7: Limit the EEG data range based on video start time ===
    ============================================================== """
    start_timestamp_sec = start_timestamp/1000
    end_timestamp_sec = end_timestamp/1000
    eeg_df = pd.read_csv(eeg_path)
    eeg_df.rename(columns={'ch1':'TP9', 'ch2':'AF7', 'ch3':'AF8', 'ch4':'TP10', 'ch5':'AUX'}, inplace=True)
    eeg_df = eeg_df.drop_duplicates()
    eeg_df = eeg_df[(eeg_df['unix_ts'] >= start_timestamp_sec) & (eeg_df['unix_ts'] <= (end_timestamp_sec))]
    eeg_df['unix_rel_ts'] = eeg_df['unix_ts'] - start_timestamp_sec
    eeg_df.to_csv(output_trimmed_csv, index=False)
    
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
    if verbose:
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
    
    fig, ax = plt.subplots(1,1,figsize=(10,4))

    psd_filedir = os.path.join(output_dir, 'temp_psd_frames')
    if not os.path.exists(psd_filedir): os.makedirs(psd_filedir)
    
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
    for eeg_current_start in frame_timestamps_list:
        eeg_current_end = eeg_current_start + 2.0
        if eeg_current_end > frame_timestamps_list[-1]: break
        psd = mne_info.compute_psd(
            tmin=eeg_current_start, 
            tmax=eeg_current_end, 
            average='mean', 
            fmin=0.5,
            fmax=60,
            verbose=False)
        powers, freqs = psd.get_data(picks=["AF7", "AF8"], return_freqs=True)
        # Note: freqs is the same size as the 2D layer of `powers`. `powers`' first dimension is for each frequency channel
        # To process, we need to look at the 2nd layer of `powers` when mapping frequencies to powers
        peak_freqs = {}
        peak_powers = {}
        # frequencies = ["delta", "theta", "alpha", "beta", "gamma"]
        for freq in frequencies:
            peak_freqs[freq] = []
            peak_powers[freq] = []
        if len(powers) > 0:
            # get through 1st layer of `powers`
            powers_avg = np.mean(powers, axis=0)
            peaks = thresholding_algo(powers_avg, 5, 3.5, 0.5)
            ax.cla()
            plt.title("Power Spectral Density\n[dt: 2]")
            ax.set_ylim([0.0, 200.0])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (Decibels)")
            for f in frequencies:
                plt.axvspan(frequency_bands[f]["range"][0], frequency_bands[f]["range"][1], color=frequency_bands[f]["color"], alpha=0.1)
            plt.plot(freqs, powers_avg, label='psd', c='b')
            plt.plot(freqs, peaks['signals'], label='peaks', c='r')
            psd_filepath = os.path.join(psd_filedir, f'frame_{eeg_current_start}.png')
            plt.savefig(psd_filepath, bbox_inches="tight")
    
    if verbose: print("Generating video from frames...")
    # Grab all frames in our temp folder. Sort them humanly.
    frames_raw = [img for img in os.listdir(psd_filedir) if img.endswith(".png")]
    frames = sort_nicely(frames_raw)
    # We'll get the first frame and temporarily use it to determine the 
    frame = cv2.imread(os.path.join(psd_filedir, frames[0]))
    height, width, layers = frame.shape
    # Initialize the video writer
    
    video = cv2.VideoWriter(psd_videopath, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    # Iterate throgh our frames
    try:
        for i in range(len(frames)):
            image = frames[i]
            video.write(cv2.imread(os.path.join(psd_filedir, image)))
    except:
        print("[ERROR] Something went wrong while processing the video. Ending early")
    # Release the video writer
    video.release()
    
    if verbose: print("Video generated. Now deleting extraneous frames from temp folder")
    shutil.rmtree(psd_filedir)
    
    if verbose: print("Video finished generating!")
    plt.close()

    return output_trimmed_csv, psd_videopath