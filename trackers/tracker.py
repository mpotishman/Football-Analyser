# trackers.py — takes raw frames from main.py, runs them through YOLO to detect objects, then uses ByteTrack
# to assign a consistent ID to each object across frames so the same player keeps the same ID throughout the video

# this section first initialises itself as a class, then detects every i frames, in this case 20 frame gap and stores it in detections variable
# the second function takes in a group of frames and returns a dictionary in the format of:
#                  tracks = {type of object: {frame number: [x1, y1, x2, y2]}}, where type of object is either player, ref or ball, and [x1, y1, x2, y2] is the bounding box for that track id


from utils import get_center_of_bbox, get_bbox_width
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import pandas as pd
import cv2
import numpy as np
sys.path.append('../')


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # ByteTrack is a tracking algorithm — it links detections across frames so each player gets a persistent ID
        self.tracker = sv.ByteTrack()
        
        def interpolate_ball_positions(self, ball_positions):
            # get the bbox values of the ball x within ball positions dictionary
            ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
            ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
            
            # interpolate missing values
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()
            
            ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
            
            return ball_positions

    # Passes every frame through the model, which returns bounding boxes + class labels (e.g player, ball) for every object it spots in each frame
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # slices frames into chunks of 20 e.g. [0:20], [20:40] to avoid memory overload
            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    # Converts raw detections into structured tracks — each object gets a class and a persistent ID across frames
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # if the code after this has already ran it should be saved in thr stub path location, so if it has been ran then just return whats already been saved rather than running it again
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # get the needed frames, so initially pass an entire video, then this only gets the every i'th frame
        detections = self.detect_frames(frames)

        tracks = {
            # eg [{0:{bbox: [0,0,0,0]}}] etc for one frame so it does it for all the frames in the video
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            # {0: 'player', 1: 'goalkeeper'} — model's index to label mapping
            cls_names = detection.names
            # flipped to {'player': 0, 'goalkeeper': 1} for easy lookup by name
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # YOLO returns its own format — this converts it to supervision's format so ByteTrack can process it
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper object to player_object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Tracks objects - adds a tracker object to the detection, now each bounding box has its own id and remains the same throughout the video
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            # define each key in the dict with another dict, then append the class id adn the position so we know where each object is at any given frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                # get the bounding box of the object, frame_detection[0] is the xyxy coordintes of each bounding box it finds, mask is 1, confidence is 2, class id is 3
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # add tracking to ref and players, not ball since its only one
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # now just the ball, since theres no need to track an id to it since its only one ball, so can hardcode track id as 1
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)

        # returns a list of dictionaries, with the positions of each bounding box at a certain frame
        return tracks

    # define ellipse drawing
    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        height = bbox[3] - bbox[1]
        width = max(24, min(int(height * 0.4), 64))



        cv2.ellipse(frame,
                    (x_center, y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=colour,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )

        # make the rectangle ast the bottom of each object
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          colour,
                          cv2.FILLED)

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f'{track_id}',
                        (int(x1_text), int(y1_rect) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6,
                        (0,0,0),
                        2)
            

        return frame

    def draw_triangle(self, frame, bbox, colour):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, colour, cv2.FILLED)
        # border
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        
        return frame
        
    # Create easier to see bounding boxes

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw players, right now player_dict = {track_id: bbox} - track id is passed as wikk be used to show each different player ID
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0,0,255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], (colour), track_id)

            # draw referees,
            for track_id, ref in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, ref["bbox"], (0, 255, 255))
                
            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(
                    frame, ball["bbox"], (0, 255, 0))
                

            output_video_frames.append(frame)

        return output_video_frames
