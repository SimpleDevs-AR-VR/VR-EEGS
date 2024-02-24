from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import argparse
import os
import cv2
import pandas as pd
import torch

def check_cuda_available():
    return torch.cuda.is_available()

def predict(input, model, use_gpu=False, preview=False):

    # Get the basename - it'll be important for saving predictions
    pred_dir = os.path.dirname(input)
    pred_dirname = os.path.splitext(os.path.basename(input))[0]
    pred_output_dir = os.path.join(pred_dir, pred_dirname)
    output_counter = 1
    while os.path.exists(pred_output_dir):
        pred_output_dir = os.path.join(pred_dir, f'{pred_dirname}{output_counter}')
        output_counter += 1
    os.makedirs(pred_output_dir)

    # Define the model, make predictions
    # However, we'll do this a BIT differently from the normal route.
    # Remember: we want to track the time passed as well.
    # So at this point, we'll have to MANUALLY iterate through each frame, using FPS and frame count to understand which millisecond we're dealing with from the video.

    # First, let's define the model itself
    model = YOLO(model)

    # Second, let's define the video capture from OpenCV
    cap = cv2.VideoCapture(input)
    frame_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    output_vidname = os.path.join(pred_output_dir,'predict.avi')
    output = cv2.VideoWriter(output_vidname,cv2.VideoWriter_fourcc(*'MJPG'),fps,(int(width),int(height)))
    
    if preview:
        output_window = 'Yolov8 frame'
        cv2.namedWindow(output_window)
    else:
        output_window = None

    # Thirdly, let's define a track_history
    predict_history = []

    # Fourthly, let's iterate through the capture
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Only continue if successful
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            if use_gpu:
                results = model.predict(frame, save=True, workers=0)
            else:
                results = model.predict(frame, save=True)
            # Get the timestamp of this frame
            frame_timestamp = float(frame_counter / fps)
            # Get the boxes and track IDs
            boxes = results[0].boxes.cpu().numpy()
            #track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            if output_window is not None:
                cv2.imshow(output_window, annotated_frame)
            # Add the annotated frame to our output
            output.write(annotated_frame)
            frame_counter += 1

            # Create an entry in our pandas database regarding this event
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = int(box.conf[0]*100)
                bx = box.xywh.tolist()
                df = pd.DataFrame({'timestamp':frame_timestamp, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
                predict_history.append(df)

            # Plot the tracks
            """
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            """
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            output.write(frame)
            # Break the loop if the end of the video is reached
            break

    output.release() 
    cap.release() 
    cv2.destroyAllWindows()

    # Finally, write our prediction history into a csv file
    df = pd.concat(predict_history)
    output_filename = os.path.join(pred_output_dir,"predicted_labels.csv")
    df.to_csv(output_filename, index=False)
    return output_filename, output_vidname

if __name__ == "__main__":
    # PARSE ARGUMENTS
    # There is 1 required argument and 1 optional argument
    parser = argparse.ArgumentParser(
                        prog='Meta Quest Pro Object Detector',
                        description='This program predicts objects found in footage captured from the Meta Quest Pro.',
                        epilog='Only use after extracting footage data using scrcpy and correcting the lens distortion usinng ffmpeg')
    parser.add_argument('input',
                        help="What video, image, or folder of video and/or images should we run the prediction on?")
    parser.add_argument('-m', '--model', 
                        help="What model should we load?", 
                        default="yolov8n.pt")
    parser.add_argument('-g', '--gpu', 
                        help="Should we use the GPU?",
                        action="store_true")
    parser.add_argument('-p', '--preview', help='Should we preview the prediction?', action='store_true')
    args = parser.parse_args()

    # Perform prediction
    predict(args.input, args.model, args.gpu, args.preview)