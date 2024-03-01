import pandas as pd
import argparse
import numpy as np
import math
import os
from scipy.optimize import minimize

def cursorConfidence(cursor_csv_path, deepsort_csv_path, output_path, newMetric):

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
        print(frame)
        if ((frame in cursor['frame'].values) == False):
            continue

        cursor_row = cursor[(cursor['frame'] == frame)]

        for index, row in df.iterrows():
            if newMetric == False:
                df.loc[index, 'cursor_conf'] = Confidence(cursor_row['x'].iloc[0], cursor_row['y'].iloc[0], row['x1'], row['y1'], row['w'], row['h'])
            elif newMetric == True:
                df.loc[index, 'cursor_conf'] = newConfidence(cursor_row['x'].iloc[0], cursor_row['y'].iloc[0], row['x1'], row['y1'], row['w'], row['h'])

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
    df.to_csv(output_path, index=False)
    print('finished')

    return output_path

def Confidence(cursor_x, cursor_y, bounding_x1, bounding_y1, w, h):
    bottom_right = np.array([632,672])
    diag_dist = np.linalg.norm(bottom_right)

    xc, yc = bounding_center(bounding_x1, bounding_y1, w, h)
    cursor = np.array([cursor_x, cursor_y])
    center = np.array([xc, yc])
    dist = np.linalg.norm(center - cursor)

    return (1 - (dist / diag_dist)) ** 2

def newConfidence(cursor_x, cursor_y, bounding_x1, bounding_y1, w, h):
    conf = 0
    xc, yc = bounding_center(bounding_x1, bounding_y1, w, h)
    pos = np.array([cursor_x, cursor_y])
    center = np.array([xc, yc])
    top_left = np.array([bounding_x1, bounding_y1])
    top_right = np.array([bounding_x1 + w, bounding_y1])
    bottom_left = np.array([bounding_x1, bounding_y1 + h])
    bottom_right = np.array([bounding_x1 + w, bounding_y1 + h])
    diag_dist_from_center = np.linalg.norm(bottom_right - center)
    
    if (h >= 1.8 * w):
        target_values = [0.8, 0.85, 0.9, 0.999]
        distances = [diag_dist_from_center, h / 2, w / 2, 1]
        c, v = compute_constants(inv_log, target_values, distances)
        max_diag = inv_log(c, v, diag_dist_from_center)
        max_hor = inv_log(c, v, w / 2)
        max_vert = inv_log(c, v, h / 2)

        target_hor = [0.9 * max_hor, 0.7 * max_hor, 0.5 * max_hor, 0.3 * max_hor, max_hor]
        distances_hor = [50, 100, 200, 400, 600, 1]
        c_hor, v_hor = compute_constants(expo, target_hor, distances_hor)

        target_vert = [0.8 * max_vert, 0.6 * max_vert, 0.4 * max_vert, 0.2 * max_vert, max_vert]
        distances_vert = [50, 100, 200, 400, 600, 1]
        c_vert, v_vert = compute_constants(expo, target_vert, distances_vert)

        if ((cursor_x > top_right[0] and cursor_y < top_right[1]) or (cursor_x < top_left[0] and cursor_y < top_left[1])
            or (cursor_x < bottom_left[0] and cursor_y > bottom_left[1]) or (cursor_x > bottom_right[0] and cursor_y > bottom_right[1])):
            x = min(abs(cursor_x - top_right[0]), abs(cursor_x - top_left[0]), abs(cursor_x - bottom_left[0]), abs(cursor_x - bottom_right[0]))
            y = min(abs(cursor_y - top_right[1]), abs(cursor_y - top_left[1]), abs(cursor_y - bottom_left[1]), abs(cursor_y - bottom_right[1]))
            conf = max_diag * np.exp(v_hor * x + v_vert * y)

        elif ((cursor_x > xc + w/2) or (cursor_x < xc - w/2)):
            x = min(abs(cursor_x - (xc + w/2)), abs(cursor_x - (xc - w/2)))
            conf = expo(c_hor, v_hor, x)
        
        elif ((cursor_y < yc - h/2) or (cursor_y > yc + h/2)):
            y = min(abs(cursor_y - (yc - h/2)), abs(cursor_y - (yc + h/2)))
            conf = expo(c_vert, v_vert, y)
        
        else:
            conf = inv_log(c, v, np.linalg.norm(pos - center))
    
    elif (w >= 1.8 * h):
        target_values = [0.8, 0.85, 0.9, 0.999]
        distances = [diag_dist_from_center, w / 2, h / 2, 1]
        c, v = compute_constants(inv_log, target_values, distances)
        max_diag = inv_log(c, v, diag_dist_from_center)
        max_hor = inv_log(c, v, w / 2)
        max_vert = inv_log(c, v, h / 2)

        target_hor = [0.8 * max_hor, 0.6 * max_hor, 0.4 * max_hor, 0.2 * max_hor, max_hor]
        distances_hor = [50, 100, 200, 400, 600, 1]
        c_hor, v_hor = compute_constants(expo, target_hor, distances_hor)

        target_vert = [0.9 * max_vert, 0.7 * max_vert, 0.5 * max_vert, 0.3 * max_vert, max_vert]
        distances_vert = [50, 100, 200, 400, 600, 1]
        c_vert, v_vert = compute_constants(expo, target_vert, distances_vert)

        if ((cursor_x > top_right[0] and cursor_y < top_right[1]) or (cursor_x < top_left[0] and cursor_y < top_left[1])
            or (cursor_x < bottom_left[0] and cursor_y > bottom_left[1]) or (cursor_x > bottom_right[0] and cursor_y > bottom_right[1])):
            x = min(abs(cursor_x - top_right[0]), abs(cursor_x - top_left[0]), abs(cursor_x - bottom_left[0]), abs(cursor_x - bottom_right[0]))
            y = min(abs(cursor_y - top_right[1]), abs(cursor_y - top_left[1]), abs(cursor_y - bottom_left[1]), abs(cursor_y - bottom_right[1]))
            conf = max_diag * np.exp(v_hor * x + v_vert * y)

        elif ((cursor_x > xc + w/2) or (cursor_x < xc - w/2)):
            x = min(abs(cursor_x - (xc + w/2)), abs(cursor_x - (xc - w/2)))
            conf = expo(c_hor, v_hor, x)
        
        elif ((cursor_y < yc - h/2) or (cursor_y > yc + h/2)):
            y = min(abs(cursor_y - (yc - h/2)), abs(cursor_y - (yc + h/2)))
            conf = expo(c_vert, v_vert, y)
        
        else:
            conf = inv_log(c, v, np.linalg.norm(pos - center))
    
    else:
        target_values = [0.8, 0.9, 0.9, 0.999]
        distances = [diag_dist_from_center, w / 2, h / 2, 1]
        c, v = compute_constants(inv_log, target_values, distances)
        max_diag = inv_log(c, v, diag_dist_from_center)
        max_hor = inv_log(c, v, w / 2)
        max_vert = inv_log(c, v, h / 2)

        target_hor = [0.9 * max_hor, 0.7 * max_hor, 0.5 * max_hor, 0.3 * max_hor, max_hor]
        distances_hor = [50, 100, 200, 400, 600, 1]
        c_hor, v_hor = compute_constants(expo, target_hor, distances_hor)

        target_vert = [0.9 * max_vert, 0.7 * max_vert, 0.5 * max_vert, 0.3 * max_vert, max_vert]
        distances_vert = [50, 100, 200, 400, 600, 1]
        c_vert, v_vert = compute_constants(expo, target_vert, distances_vert)

        if ((cursor_x > top_right[0] and cursor_y < top_right[1]) or (cursor_x < top_left[0] and cursor_y < top_left[1])
            or (cursor_x < bottom_left[0] and cursor_y > bottom_left[1]) or (cursor_x > bottom_right[0] and cursor_y > bottom_right[1])):
            x = min(abs(cursor_x - top_right[0]), abs(cursor_x - top_left[0]), abs(cursor_x - bottom_left[0]), abs(cursor_x - bottom_right[0]))
            y = min(abs(cursor_y - top_right[1]), abs(cursor_y - top_left[1]), abs(cursor_y - bottom_left[1]), abs(cursor_y - bottom_right[1]))
            conf = max_diag * np.exp(v_hor * x + v_vert * y)

        elif ((cursor_x > xc + w/2) or (cursor_x < xc - w/2)):
            x = min(abs(cursor_x - (xc + w/2)), abs(cursor_x - (xc - w/2)))
            conf = expo(c_hor, v_hor, x)
        
        elif ((cursor_y < yc - h/2) or (cursor_y > yc + h/2)):
            y = min(abs(cursor_y - (yc - h/2)), abs(cursor_y - (yc + h/2)))
            conf = expo(c_vert, v_vert, y)
        
        else:
            conf = inv_log(c, v, np.linalg.norm(pos - center))
        
    return conf


def compute_constants(func, targets : list, dists : list):
    def cost_function(params):
        c_val, v_val = params
        function_values = [func(c_val, v_val, distance) for distance in dists]
        differences = [abs(function_value - target_value) for function_value, target_value in zip(function_values, targets)]
        return sum(differences)

    guess = [1, 1]
    result = minimize(cost_function, guess, method='Nelder-Mead')

    return result.x[0], result.x[1]

    

def bounding_center(x1, y1, w, h):
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    return x, y

def expo(c, v, x):
    return c * np.exp(v * x)

def inv_log(c, v, dist):
    return c / np.log(v + dist)


if __name__ == "__main__":
    """ parser = argparse.ArgumentParser()
    parser.add_argument('cursor',help='Eye Cursor csv file')
    parser.add_argument('deepsort',help='Deepsort Tracks csv file')
    args = parser.parse_args() """

    cursorConfidence('sample/deepsort_predict_eycursor.csv', 'sample/deepsort_tracks.csv', 'sample/deepsort_cursorConfidence_new.csv', True)