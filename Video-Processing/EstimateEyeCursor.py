import json
import argparse
import pandas as pd
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from query_yes_no import query_yes_no

def EstimateCursor(
        source_filepath, 
        events_filepath, 
        mapping_filepath, 
        frame_timestamps_filepath,
        start_timestamp=None,
        end_timestamp=None,
        force_overwrite=False
            ):
    # Note: `sim_start_ts` is expected to be in unix seconds

    """ =================================
    === Step 1: Read the mapping data ===
    ================================= """
    with open(mapping_filepath) as jsonfile:
        mapping = json.load(jsonfile)

    """ ===============================================================
    === Step 2: Get the transformation matrix from the mapping data ===
    =============================================================== """
    transformation_matrix = np.array(mapping['transformation_matrix'])
    def transform(input):
        A = np.array(input + [1])
        return np.dot(A,transformation_matrix)

    """ ================================
    === Step 3: Read the events data ===
    ================================ """
    events_df = pd.read_csv(events_filepath)
    eye_df = events_df[
        (events_df['event_type'] == 'Eye Tracking') 
        & (events_df['description'] == 'Screen Position')
        & (events_df['title'] == 'Left')
    ]
    if start_timestamp is not None: eye_df = eye_df[eye_df['unix_ms'] >= start_timestamp]
    if end_timestamp is not None: eye_df = eye_df[eye_df['unix_ms'] <= end_timestamp]
    events_start = start_timestamp if start_timestamp is not None else events_df.iloc[1]['unix_ms']
    eye_df['unix_rel'] = eye_df['unix_ms'].apply(lambda x: (x-events_start)/1000)

    """ ==========================================
    === Step 4: Read the frame timestamps data ===
    ==============================================
    Note that all the timestamps are in seconds here. Hence the /1000 division above. """
    frame_ts_df = pd.read_csv(frame_timestamps_filepath)

    """ =======================================
    === Step 5: Define the output filenames ===
    ======================================= """
    root_dir = os.path.dirname(source_filepath)
    out_filename = os.path.splitext(os.path.basename(source_filepath))[0]
    out_vidpath = os.path.join(root_dir, f'{out_filename}_eyecursor.avi')
    out_csvpath = os.path.join(root_dir, f'{out_filename}_eycursor.csv')

    produce_video = (force_overwrite or not os.path.exists(out_vidpath) or query_yes_no(f"The file \"{out_vidpath}\" already exists. Do you wish to overwrite it?", default=None))
    produce_csv = (force_overwrite or not os.path.exists(out_csvpath) or query_yes_no(f"The file \"{out_csvpath}\" already exists. Do you wish to overwrite it?", default=None))
    if not (produce_video or produce_csv):
        print("You've opted to not produc any video or CSV file. Ending early.")
        return out_vidpath, out_csvpath

    cap = cv2.VideoCapture(source_filepath)
    capw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    caph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float `height`
    capfps = int(cap.get(cv2.CAP_PROP_FPS))          # FPS
    if produce_video:
        out = cv2.VideoWriter(out_vidpath, cv2.VideoWriter_fourcc('M','J','P','G'), capfps, (capw,caph))

    if produce_csv:
        fields = ['frame', 'x', 'y']
        csvfile = open(out_csvpath, 'w')
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 

    prev_timestamp = 0.0
    frame_counter = -1
    eye_pos = None
    while(cap.isOpened()):
        success, frame = cap.read()
        if success:
            frame_counter += 1
            frame_row = frame_ts_df[frame_ts_df['frame'] == frame_counter].iloc[0]
            frame_ts = frame_row['timestamp']
            result = np.copy(frame)
            eye_positions = eye_df[(eye_df['unix_rel'] >= prev_timestamp) & (eye_df['unix_rel'] < frame_ts)]
            if not eye_positions.empty:
                ep = eye_positions.loc[eye_positions.index[0]]
                eye_pos_est = transform([ep['x'], ep['y']])
                eye_pos = (int(eye_pos_est[0]), int(caph-eye_pos_est[1]))
                # we only update the csv if the eye position is new
                if produce_csv: csvwriter.writerow([frame_counter, eye_pos[0], eye_pos[1]])
            if produce_video:
                if eye_pos is not None:
                    result = cv2.drawMarker(result, eye_pos, (255,0,0), cv2.MARKER_CROSS, 20, 2)
                out.write(result)
            prev_timestamp = frame_ts
        else:
            break

    cap.release()
    if produce_video: out.release()
    if produce_csv: csvfile.close()

    return out_vidpath, out_csvpath

def EstimateDepth(
        root,
        left_filename, 
        right_filename  
            ):
    left_filepath = os.path.join(root, left_filename)
    right_filepath = os.path.join(root, right_filename)
    out_vidpath = os.path.join(root, f'depth.avi')
    print(left_filepath, right_filepath)

    cap_left = cv2.VideoCapture(left_filepath)
    cap_right = cv2.VideoCapture(right_filepath)

    capw  = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    caph = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float `height`
    capfps = int(cap_left.get(cv2.CAP_PROP_FPS))          # FPS
    out = cv2.VideoWriter(out_vidpath, cv2.VideoWriter_fourcc('M','J','P','G'), capfps, (capw,caph))
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    while(True):
        ls, lframe = cap_left.read()
        rs, rframe = cap_right.read()
        if ls and rs:
            lfgray = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
            rfgray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(lfgray,rfgray)
            out.write(disparity)
        else:
            break

    cap_left.release()
    cap_right.release()
    out.release()

    return out_vidpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source',help='The footage that needs to have the cursor estimated on.')
    parser.add_argument('events',help='The CSV file generated from the vr simulation that contains eye positions')
    parser.add_argument('mapping',help='The mapping data that contains estimations of anchor positions')    
    parser.add_argument('frame_timestamps',help='The CSV file that contains the timestamps of each frame')
    args = parser.parse_args()

    EstimateCursor(args.source, args.events, args.mapping, args.frame_timestamps)