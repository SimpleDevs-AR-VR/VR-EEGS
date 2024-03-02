import os
import re
import shutil
import numpy as np
import argparse
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

_COL_DICT = {
    'ch1':'TP9', 
    'ch2':'AF7', 
    'ch3':'AF8', 
    'ch4':'TP10', 
    'ch5':'AUX'
}
_EEG_CHANNELS = ["AF7","AF8","TP9","TP10"]
_EEG_REF_CHANNELS = ["TP9", "TP10"]
_FREQUENCY_BANDS = {
    "delta": {"range":(0.5,4),"color":"white"},
    "theta": {"range":(4, 8),"color":"darkgrey"},
    "alpha": {"range":(8, 16),"color":"blue"},
    "beta":  {"range":(16, 32),"color":"orange"},
    "gamma": {"range":(32, 80),"color":"red"}
}

def GetMeanAbsolutePower(powers, freqs, freq_bands, bands=["theta","alpha","beta","gamma"]):
    # Get the mean absolute power of each frequency band, given the powers and frequencies of a Power Spectral Density
    mean_abs_powers = {}
    for band in bands:
        freq_range = freq_bands[band]['range']
        mean_abs_powers[band] = np.mean([powers[i] for i in range(len(freqs)) if freqs[i] >= freq_range[0] or freqs[i] <= freq_range[1]])
    return mean_abs_powers

def CreateEEGVideo(
        eeg_path, 
        eeg_column_dict,
        eeg_timestamp_colname,
        eeg_channels,
        eeg_ref_channels,

        start_timestamp = None, 
        end_timestamp = None, 
        timestamps_list = None, 

        l_freq = None,
        h_freq = None,
        notch_freqs = None,

        display_channels = ["AF7", "AF8"],
        display_xlims = [0.5,80],
        display_ylims = [0,200],
        fps=60, 
        
        output_dir=None,
        output_trimname="eeg_trim", 
        output_vidname="psd", 
        output_psdname="psd",
        
        verbose=True
            ):

    """ ===========================================
    === Step 1: Determine the output file paths ===
    =========================================== """
    if output_dir is None: output_dir = os.path.dirname(eeg_path)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_trimmed_csv = os.path.join(output_dir,f'{output_trimname}.csv')
    psd_videopath = os.path.join(output_dir, f'{output_vidname}.avi')
    psd_csvpath = os.path.join(output_dir, f'{output_psdname}.csv')


    """ ==========================================
    === Step 2: Read the original EEG raw data ===
    ========================================== """
    eeg_df = pd.read_csv(eeg_path)
    eeg_df.rename(columns=eeg_column_dict, inplace=True)
    eeg_df = eeg_df.drop_duplicates()


    """ ======================================================================
    === Step 3: Limit the EEG data range based on video start and end time ===
    ====================================================================== """
    # `start_timestamp` and `end_timestamp` are expected to be in the same timestamp as the provided timestamp column name
    if start_timestamp is not None:
        eeg_df = eeg_df[eeg_df[eeg_timestamp_colname] >= start_timestamp]
    if end_timestamp is not None: 
        eeg_df = eeg_df[eeg_df[eeg_timestamp_colname] <= end_timestamp]
    rel_start = start_timestamp if start_timestamp is not None else eeg_df.iloc[0][eeg_timestamp_colname]
    eeg_df['unix_rel_ts'] = eeg_df[eeg_timestamp_colname] - rel_start


    """ =======================================================
    === Step 4: Save the trimmed version of the raw eeg data ===
    ======================================================= """
    eeg_df.to_csv(output_trimmed_csv, index=False)
    

    """ =============================================
    === Step 5: Process EEG data into mne package ===
    ============================================= """
    eeg_start = eeg_df.iloc[0][eeg_timestamp_colname]
    eeg_end = eeg_df.iloc[-1][eeg_timestamp_colname]
    eeg_duration = eeg_end - eeg_start
    eeg_size = len(eeg_df.index)
    eeg_frequency = round(eeg_size / eeg_duration)
    eeg_info = mne.create_info(eeg_channels, eeg_frequency, ch_types='eeg', verbose=False)
    s_array = np.transpose(eeg_df[eeg_channels].to_numpy())
    mne_info = mne.io.RawArray(s_array, eeg_info, first_samp=0, copy='auto', verbose=False)
    mne_info.set_eeg_reference(ref_channels=eeg_ref_channels)
    if l_freq is not None or h_freq is not None:
        mne_info.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if notch_freqs is not None and len(notch_freqs)>0:
        mne_info.notch_filter(freqs=notch_freqs, verbose=False)
    if verbose:
        print('eeg_start: ' + str(eeg_start))
        print('eeg_end: ' + str(eeg_end))
        print('eeg_duration: ' + str(eeg_duration))

    """ ======================================
    === Step 6: Parsing EEG frame-by-frame ===
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
    if os.path.exists(psd_filedir): shutil.rmtree(psd_filedir)
    os.makedirs(psd_filedir)
    map_filedir = os.path.join(output_dir, 'temp_map_frames')
    if os.path.exists(map_filedir): shutil.rmtree(map_filedir)
    os.makedirs(map_filedir)

    if timestamps_list is None: timestamps_list = eeg_df['unix_rel_ts'].unique()
    current_frame_counter = 0
    for eeg_current_end in timestamps_list:
        eeg_current_start = eeg_current_end - 2.0
        current_frame_counter += 1
        
        if eeg_current_start < 0.0:
            ax.cla()
            plt.title("Power Spectral Density\n[dt: 2]")
            ax.set_ylim(display_ylims)
            ax.set_xlim(display_xlims)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density (Db/Hz)")
            for f in _FREQUENCY_BANDS:
                plt.axvspan(
                    _FREQUENCY_BANDS[f]["range"][0], 
                    _FREQUENCY_BANDS[f]["range"][1], 
                    color=_FREQUENCY_BANDS[f]["color"], 
                    alpha=0.1
                )
            psd_filepath = os.path.join(psd_filedir, f'frame_{current_frame_counter}.png')
            plt.savefig(psd_filepath, bbox_inches="tight")
            
            ax.cla()
            plt.title("Power Spectral Density: Mean Abs. Power\n[dt: 2]")
            ax.set_ylim(display_ylims)
            ax.set_xlim(display_xlims)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density: Mean Abs. Power (Db/Hz)")
            for f in _FREQUENCY_BANDS:
                plt.axvspan(
                    _FREQUENCY_BANDS[f]["range"][0], 
                    _FREQUENCY_BANDS[f]["range"][1], 
                    color=_FREQUENCY_BANDS[f]["color"], 
                    alpha=0.1
                )
            map_filepath = os.path.join(map_filedir, f'frame_{current_frame_counter}.png')
            plt.savefig(psd_filepath, bbox_inches="tight")
            continue
        
        psd = mne_info.compute_psd(
                tmin=eeg_current_start, 
                tmax=eeg_current_end, 
                average='mean', 
                fmin=display_xlims[0],
                fmax=display_xlims[1],
                verbose=False
                    )
        powers, freqs = psd.get_data(picks=display_channels, return_freqs=True)
        if current_frame_counter == 1 and verbose: print(powers, freqs)
        # Note: freqs is the same size as the 2D layer of `powers`. `powers`' first dimension is for each frequency channel
        # To process, we need to look at the 2nd layer of `powers` when mapping frequencies to powers
        if len(powers) == 0:
            ax.cla()
            plt.title("Power Spectral Density\n[dt: 2]")
            ax.set_ylim(display_ylims)
            ax.set_xlim(display_xlims)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density (Db/Hz)")
            for f in _FREQUENCY_BANDS:
                plt.axvspan(
                    _FREQUENCY_BANDS[f]["range"][0], 
                    _FREQUENCY_BANDS[f]["range"][1], 
                    color=_FREQUENCY_BANDS[f]["color"], 
                    alpha=0.1
                )
            psd_filepath = os.path.join(psd_filedir, f'frame_{current_frame_counter}.png')
            plt.savefig(psd_filepath, bbox_inches="tight")
            continue

        # get through 1st layer of `powers`
        powers_avg = np.mean(powers, axis=0)
        plt.plot(freqs, powers_avg, label='psd', c='b')
        psd_filepath = os.path.join(psd_filedir, f'frame_{current_frame_counter}.png')
        plt.savefig(psd_filepath, bbox_inches="tight")

        #peaks = thresholding_algo(powers_avg, 5, 3.5, 0.5)
        #plt.plot(freqs, peaks['signals'], label='peaks', c='r')
        

        
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """
    eeg_path, 
    eeg_column_dict,
    eeg_timestamp_colname,
    eeg_channels,
    eeg_ref_channels,
    """
    parser.add_argument("eeg", 
                        help="The path to the raw EEG data")
    # `eeg_column_dict`: _COL_DICT
    parser.add_argument("eeg_timestamp_colname",
                        help="The name of the column that is meant to represent the timestamp column")
    # `eeg_channels`: _EEG_CHANNELS
    # `eeg_ref_channels`: _EEG_REF_CHANNELS

    """
    start_timestamp, 
    end_timestamp, 
    timestamps_list = None, 
    """
    parser.add_argument("-st", "--start_timestamp", 
                        help="The unix start timestamp where we want to restrict the EEG to. Must match the timescale of the provided `eeg_timestamp_colname` (ex. if `eeg_timestamp_colname` is in seconds, this must also be in seconds).",
                        type=float,
                        default=None )
    parser.add_argument("-et", "--end_timestamp", 
                        help="The unix end timestamp where we want to restrict the EEG to. Must match the timescale of the provided `eeg_timestamp_colname` (ex. if `eeg_timestamp_colname` is in seconds, this must also be in seconds).",
                        type=float,
                        default=None )
    parser.add_argument("-tlf", "--timestamp_list_file", 
                        help="The relative path to a csv file that contains which timestamps we want to look at. Only works if `-tc` is set.", 
                        default=None )
    parser.add_argument("-tc", '--timestamp_list_column', 
                        help="If we decide to read timestamps from `-tlf`, what column represents the timestamps? Must be a seconds-based timestamp column. Only works if `-tlf` is set.", 
                        default=None )

    """
    l_freq = None,
    h_freq = None,
    notch_freqs = None,
    """
    parser.add_argument("-lf", "--l_freq", 
                        help="The lower frequency we want to filter. If not provided, the resulting filter will become a low-pass filter.", 
                        type=float, 
                        default=0.5 )
    parser.add_argument('-hf', '--h_freq', 
                        help="The upper frequency we want to filter. If not provided, the resulting filter will become a high-pass filter.", 
                        type=float, 
                        default=60.0 )
    parser.add_argument("-nf", "--notch_freqs",
                        help="If a notch filter is needed, provide the necessary frequencies to filter out.",
                        nargs="+",
                        type=float,
                        default=None)

    """
    display_channels = ["AF7", "AF8"],
    display_xlims = [0.5,80],
    display_ylims = [0,200],
    fps=60,
    """
    parser.add_argument("-dc", "--display_channels",
                        help="The channels we want to restrict the visualization to.",
                        nargs="+",
                        default=["AF7","AF8"] )
    parser.add_argument("-dlx", '--display_l_x', 
                        help="The lower frequency we want to restrict the visualization to.", 
                        type=float, 
                        default=0.5)
    parser.add_argument("-dhx", '--display_h_x', 
                        help="The upper frequency we want to restrict the visualization to.", 
                        type=float, 
                        default=60.0 )
    parser.add_argument("-dly", '--display_l_y', 
                        help="The lower power we want to restrict the visualization to.", 
                        type=float, 
                        default=0.0 )
    parser.add_argument("-dhy", '--display_h_y', 
                        help="The upper power we want to restrict the visualization to.", 
                        type=float, 
                        default=200.0 )
    parser.add_argument('-fps', '--frames_per_second', 
                        help="The frames per second we want to set the video to.", 
                        type=float, 
                        default=60 )

    """
    output_dir="./",
    output_trimname="eeg_trim", 
    output_vidname="psd", 
    output_psdname="psd",
    """
    parser.add_argument("-od", "--out_dir", 
                        help="The directory where we want to save the results in.",
                        default=None )
    parser.add_argument("-ot", '--out_trimname', 
                        help="The name (no extension) of the outputted csv file after calculating the PSD for each frame.", 
                        default="eeg_trim" )
    parser.add_argument("-ov", '--out_vidname', 
                        help="The name (no extension) of the outputted video file.", 
                        default="psd" )
    parser.add_argument('-op', '--out_psdname', 
                        help="The name (no extension) of the outputted psd values file.",
                        default="psd" )

    parser.add_argument("-v", "--verbose",
                        help="Should we be verbose in our messaging?",
                        action="store_true")

    args = parser.parse_args()

    timestamp_list = None
    if args.timestamp_list_file is not None and args.timestamp_list_column is not None:
        timestamp_list_df = pd.read_csv(args.timestamp_list_file)
        timestamp_list = timestamp_list_df[args.timestamp_list_column].to_list()
    
    CreateEEGVideo(
        args.eeg,
        _COL_DICT,
        args.eeg_timestamp_colname,
        _EEG_CHANNELS,
        _EEG_REF_CHANNELS,
        args.start_timestamp,
        args.end_timestamp,
        timestamps_list=timestamp_list, 
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freqs=args.notch_freqs,
        display_channels=args.display_channels,
        display_xlims=[args.display_l_x, args.display_h_x],
        display_ylims =[args.display_l_y,args.display_h_y],
        fps=args.frames_per_second, 
        output_dir=args.out_dir,
        output_trimname=args.out_trimname, 
        output_vidname=args.out_vidname, 
        output_psdname=args.out_psdname,
        verbose=args.verbose
    )