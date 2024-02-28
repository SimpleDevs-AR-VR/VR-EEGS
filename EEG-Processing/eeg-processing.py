import pandas as pd
import numpy as np
import mne
import argparse
import sys

class EEGProcessor:
    def __init__(self, src, channel_dict=None):
        self.change_src(src, channel_dict)
    
    def change_src(self, new_src, channel_dict=None):
        self.src = new_src
        self.raw_df = pd.read_csv(self.src)
        self.raw_df = self.raw_df.drop_duplicates()
        self.mne = None
        if channel_dict is not None: self.set_channels(channel_dict)
        else: self.channel_dict = None
        return self
    def reload_src(self):
        self.raw_df = pd.read_csv(self.src)
        self.raw_df = self.raw_df.drop_duplicates()
        self.mne = None
        if self.channel_dict is not None: self.set_channels(self.channel_dict)
        return self
    def set_channels(self, channel_dict={'ch1':'TP9', 'ch2':'AF7', 'ch3':'AF8', 'ch4':'TP10', 'ch5':'AUX'}):
        self.channel_dict = channel_dict
        self.raw_df.rename(columns=channel_dict, inplace=True)
        return self

    def filter_timestamps(self, timestamp_col, min_start=None, max_end=None, filter_out_range=None):
        if min_start is not None:
            self.raw_df = self.raw_df[self.raw_df[timestamp_col] >= min_start]
        if max_end is not None:
            self.raw_df = self.raw_df[self.raw_df[timestamp_col] <= max_end]
        if filter_out_range is not None:
            self.raw_df = self.raw_df[(self.raw_df[timestamp_col] >= filter_out_range[1]) & (self.raw_df[timestamp_col] <= filter_out_range[0])]
        self.mne = None
        return self
    
    def generate_mne(self, timestamp_col, electrodes=["TP9","TP10","AF7", "AF8"], verbose=False):
        eeg_start = self.raw_df.iloc[0][timestamp_col]
        eeg_end = self.raw_df.iloc[-1][timestamp_col]
        eeg_duration = eeg_end - eeg_start
        eeg_size = len(self.raw_df.index)
        eeg_frequency = round(eeg_size / eeg_duration)
        if verbose:
            print('eeg_start: ' + str(eeg_start))
            print('eeg_end: ' + str(eeg_end))
            print('eeg_duration: ' + str(eeg_duration))

        eeg_info = mne.create_info(electrodes, eeg_frequency, ch_types='eeg', verbose=False)
        s_array = np.transpose(self.raw_df[electrodes].to_numpy())
        self.mne = mne.io.RawArray(s_array, eeg_info, first_samp=0, copy='auto', verbose=False)
        return self
    def rereference(self, new_refs=['TP9', 'TP10']):
        self.mne.set_eeg_reference(ref_channels=new_refs)
    def low_pass(self, freq):
        if self.mne is None: 
            print("ERROR: No MNE has been generated yet.")
            return self
        self.mne.filter(l_freq=None, h_freq=freq, verbose=False)
        return self
    def high_pass(self, freq):
        if self.mne is None:
            print("ERROR: No MNE has been generated yet.")
            return self
        self.mne.filter(l_freq=freq, h_freq=None, verbose=False)
        return self
    def bandass_filter(self, l_freq, h_freq):
        if self.mne is None:
            print("ERROR: No MNE has been generated yet.")
            return self
        self.mne.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        return self
    def notch_filter(self, freqs):
        if self.mne is None:
            print("ERROR: No MNE has been generated yet.")
            return self
        self.mne.notch_filter(freqs=freqs, verbose=None)
        return self
    
    def plot_mne(self, start=0, duration=10.0, picks=['AF7', 'AF8'], scaling="auto"):
        if self.mne is None:
            print("ERROR: No MNE has been generated yet.")
            return self
        self.mne.plot(start=start, duration=duration, scalings=scaling)


    
    def export_raw(self, output_filename):
        self.raw_df.to_csv(output_filename, index=False)
        return self
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="The relative path to the eeg data file")
    parser.add_argument("ts_col", help="What's the primary timestamp column in our data?")
    parser.add_argument("-st", "--start_timestamp", 
                        help="If we want to restrict the timescale of our data, what's the minimum timestamp to clamp?", 
                        type=float, default=None)
    parser.add_argument("-et", "--end_timestamp",
                        help="If we want to restrict the timescale of our data, what's the maximum timestamp to clamp?",
                        type=float, default=None)
    parser.add_argument("-fc",'--frequency_channels', 
                        help="To which frequencies do we want to pay attention to when generating the MNE?",
                        nargs='+', default=["TP9","TP10","AF7", "AF8"])
    parser.add_argument("-rf", '--reference_channels',
                        help="If we want to re-reference the data, to which frequencies do we re-reference?",
                        nargs="+", default=["TP9", "TP10"])
    parser.add_argument('-lf', '--lowest_frequency', 
                        help="If we want to filter the frequencies, what's the lowest frequency we'll allow?",
                        type=float, default=None)
    parser.add_argument('-hf', '--highest_frequency',
                        help="If we want to filter the frequencies, what's the biggest frequency we'll allow?",
                        type=float, default=None)
    parser.add_argument('-nf', '--notch_frequencies', 
                        help="If we want to notch filter, what are those frequencies?",
                        nargs="+", type=float, default=None)
    parser.add_argument("-er", '--export_raw', 
                        help="Should we export the end-result raw eeg data into a csv? If so, provide the filename",
                        nargs="?", default=None)
    args = parser.parse_args()

    channel_dict={'ch1':'TP9', 'ch2':'AF7', 'ch3':'AF8', 'ch4':'TP10', 'ch5':'AUX'}
    processor = EEGProcessor(args.src, channel_dict=channel_dict)
    if args.start_timestamp is not None or args.end_timestamp is not None:
        processor.filter_timestamps(args.ts_col, args.start_timestamp, args.end_timestamp)
    if args.export_raw is not None:
        processor.export_raw(args.export_raw)

    processor.generate_mne(args.ts_col, args.frequency_channels)
    processor.rereference(args.reference_channels)
    if args.lowest_frequency is not None or args.highest_frequency is not None:
        processor.bandass_filter(args.lowest_frequency, args.highest_frequency)
    if args.notch_frequencies is not None:
        processor.notch_filter(args.notch_frequencies)

    processor.plot_mne()
    while True:
        user_input = input().lower()
        if user_input == "q":
            break
    