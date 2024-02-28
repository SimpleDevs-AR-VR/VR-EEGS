import os
from ultralytics import YOLO
import cv2
import cvzone
import pandas as pd
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from query_yes_no import query_yes_no

def check_cuda_available():
    return torch.cuda.is_available()

class ObjectDetection():

    def __init__(self, model, conf_thresh=0.5):
        self.model = self.load_model(model)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf_thresh = conf_thresh

    def load_model(self, model):
        _model = YOLO(model)
        _model.fuse()
        return _model
    
    def predict(self, frame, use_gpu=False):
        if use_gpu:
            results = self.model(frame, stream=True, workers=0)
        else:
            results = self.model(frame, stream=True)
        return results
    
    def plot_boxes(self, results, frame):
        detections = []
        for r in results:
            for box in r.boxes:
                # generating the bounding box
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = abs(x2-x1), abs(y2-y1)

                # dealing with the classname
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                # dealing with confidence score
                conf = math.ceil(box.conf[0]*100)/100
                if conf > self.conf_thresh:
                    detections.append((([x1,y1,w,h]), conf, currentClass))

        return detections, frame
    
    def track_detect(self, detections, frame, tracker, frame_id):
        tracks = tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb
            x1,y1,x2,y2 = bbox
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = abs(x2-x1), abs(y2-y1)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))
            cvzone.cornerRect(frame, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))
            df = pd.DataFrame({'frame_id':frame_id, 'track_id':track_id, 'track_class':track.det_class, 'track_conf': track.det_conf, 'x1':x1, 'y1':y1, 'w':w, 'h':h}, index=[0])
            active_tracks.append(df)
        return frame, active_tracks
    
    def __call__(self, src, output_dir, use_gpu=False, preview=False, force_overwrite=False):
        tracker = DeepSort(max_age=5,
                           n_init=2,
                           nms_max_overlap=1.0,
                           max_cosine_distance=0.3,
                           nn_budget=None,
                           override_track_class=None,
                           embedder="mobilenet",
                           half=True,
                           bgr=True,
                           embedder_gpu=True,
                           embedder_model_name=None,
                           embedder_wts=None,
                           polygon=False,
                           today=None)

        cap = cv2.VideoCapture(src)
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        output_vidpath = os.path.join(output_dir,'deepsort_predict.avi')
        output_filepath = os.path.join(output_dir,"deepsort_tracks.csv")
        produce_video = (force_overwrite or not os.path.exists(output_vidpath) or query_yes_no(f"The file \"{output_vidpath}\" already exists. Do you wish to overwrite it?", default=None))
        produce_csv = (force_overwrite or not os.path.exists(output_filepath) or query_yes_no(f"The file \"{output_filepath}\" already exists. Do you wish to overwrite it?", default=None))
        if not (produce_video or produce_csv):
            print("Opted out of producing deepsort calcualtions. Ending early.")
            return output_filepath, output_vidpath
        
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if produce_video: output_vid = cv2.VideoWriter(output_vidpath, cv2.VideoWriter_fourcc(*'MJPG'),fps,(int(w),int(h)))
        if produce_csv: track_history = []
        frame_counter = -1

        while True:
            success, frame = cap.read()
            if not success: break
            frame_counter += 1
            results = self.predict(frame, use_gpu)
            detections, _frame = self.plot_boxes(results, frame)
            detect_frame, detect_tracks = self.track_detect(detections, _frame, tracker, frame_counter)
            if produce_csv: track_history.extend(detect_tracks)
            if produce_video: output_vid.write(detect_frame)
            if preview: cv2.imshow('Image', detect_frame)
            #if cv2.waitKey(1) == ord('q'): break

        cap.release()
        if produce_video: output_vid.release()
        if preview: cv2.destroyAllWindows()
        if produce_csv:
            df = pd.concat(track_history)
            df.to_csv(output_filepath, index=False)

        return output_filepath, output_vidpath

