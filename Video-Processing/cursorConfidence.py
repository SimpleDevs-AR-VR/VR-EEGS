import pandas as pd
import argparse
import numpy as np
import math

def cursorConfidence(cursor_csv_path, deepsort_csv_path):
    file_path = 'sample/deepsort_cursorConfidence.csv'

    confidence = {'frame_id' : [], 
                  'track_id' : [],
                  'class_name' : [],
                  'cursor_confidence' : [], 
                  'deepsort_confidence' : [],
                  'x1' : [],
                  'y1' : [],
                  'w' : [],
                  'h' : []
                  }


    cursor = pd.read_csv(cursor_csv_path)
    deepsort = pd.read_csv(deepsort_csv_path)

    deepsort_sorted = deepsort.groupby(deepsort['frame_id'])
    deepsort_dict = {k: v for k, v in deepsort_sorted}

    for key, df in deepsort_dict.items():
        frame = key
        if ((frame in cursor['frame'].values) == False):
            continue

        cursor_row = cursor[(cursor['frame'] == frame)]

        for index, row in df.iterrows():
            df.loc[index, 'cursor_conf'] = Confidence(cursor_row['x'].iloc[0], cursor_row['y'].iloc[0], row['x1'], row['y1'], row['w'], row['h'])

        conf_sorted = df.sort_values(by='cursor_conf', ascending=False)

        for index, row in conf_sorted.iterrows():
            confidence['frame_id'].append(frame)
            confidence['track_id'].append(row['track_id'])
            confidence['class_name'].append(row['track_class'])
            confidence['cursor_confidence'].append(row['cursor_conf'])
            confidence['deepsort_confidence'].append(row['track_conf'])
            confidence['x1'].append(row['x1'])
            confidence['y1'].append(row['y1'])
            confidence['w'].append(row['w'])
            confidence['h'].append(row['h'])

    df = pd.DataFrame(confidence)
    df.to_csv(file_path, index=False)
    print('finished')

    return file_path

def Confidence(cursor_x, cursor_y, bounding_x1, bounding_y1, w, h):
    bottom_right = np.array([632,672])
    diag_dist = np.linalg.norm(bottom_right)

    xc, yc = bounding_center(bounding_x1, bounding_y1, w, h)
    cursor = np.array([cursor_x, cursor_y])
    center = np.array([xc, yc])
    dist = np.linalg.norm(center - cursor)

    return (1 - (dist / diag_dist)) ** 2
    

def bounding_center(x1, y1, w, h):
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    return x, y

if __name__ == "__main__":
    """ parser = argparse.ArgumentParser()
    parser.add_argument('cursor',help='Eye Cursor csv file')
    parser.add_argument('deepsort',help='Deepsort Tracks csv file')
    args = parser.parse_args() """

    cursorConfidence('sample/deepsort_predict_eycursor.csv', 'sample/deepsort_tracks.csv')