# trackers.py — takes raw frames from main.py, runs them through YOLO, then uses
# Ultralytics BoT-SORT/ReID to assign object IDs across frames.

# this section first initialises itself as a class, then detects every i frames, in this case 20 frame gap and stores it in detections variable
# the second function takes in a group of frames and returns a dictionary in the format of:
#                  tracks = {type of object: {frame number: [x1, y1, x2, y2]}}, where type of object is either player, ref or ball, and [x1, y1, x2, y2] is the bounding box for that track id


from utils import get_center_of_bbox, get_bbox_width
from ultralytics import YOLO
import pickle
import os
import sys
import pandas as pd
import cv2
import numpy as np
sys.path.append('../')


class Tracker:
    def __init__(self, model_path, tracker_config="trackers/botsort_reid.yaml"):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        
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

    # Converts raw detections into structured tracks.
    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        tracker_config=None
    ):

        # if the code after this has already ran it should be saved in thr stub path location, so if it has been ran then just return whats already been saved rather than running it again
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracker_config = tracker_config or self.tracker_config
        previous_source_frame_num = None
        track_id_offset = 0
        max_track_id_seen = 0

        tracks = {
            # eg [{0:{bbox: [0,0,0,0]}}] etc for one frame so it does it for all the frames in the video
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, frame in enumerate(frames):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            source_frame_num = self.get_source_frame_number(frames, frame_num)
            if self.should_reset_tracker(frame_num, source_frame_num, previous_source_frame_num):
                if frame_num != 0:
                    track_id_offset = max_track_id_seen
                    print(
                        "Resetting tracker before filtered frame "
                        f"{frame_num} (source frame jumped from "
                        f"{previous_source_frame_num} to {source_frame_num})"
                    )
                self.reset_ultralytics_tracker()

            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            if result.boxes is None:
                previous_source_frame_num = source_frame_num
                continue

            bboxes = result.boxes.xyxy.cpu().tolist()
            cls_ids = result.boxes.cls.cpu().int().tolist()
            track_ids = self.get_track_ids(result)

            for bbox, cls_id, track_id in zip(bboxes, cls_ids, track_ids):
                object_type = result.names[cls_id]
                if track_id is not None:
                    track_id = track_id + track_id_offset
                    max_track_id_seen = max(max_track_id_seen, track_id)

                if object_type == "goalkeeper":
                    object_type = "player"

                if object_type == "player" and track_id is not None:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                elif object_type == "referee" and track_id is not None:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                elif object_type == "ball":
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            previous_source_frame_num = source_frame_num

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        # returns a list of dictionaries, with the positions of each bounding box at a certain frame
        return tracks

    # when frames is a FrameSubset, keep track of the original video frame number
    def get_source_frame_number(self, frames, frame_num):
        if hasattr(frames, "indexes"):
            return frames.indexes[frame_num]
        return frame_num

    # reset at the first frame and whenever a filtered video skips source frames
    def should_reset_tracker(self, frame_num, source_frame_num, previous_source_frame_num):
        if frame_num == 0:
            return True

        if source_frame_num is None or previous_source_frame_num is None:
            return False

        return source_frame_num != previous_source_frame_num + 1

    # clear BoT-SORT/ByteTrack's live state without recreating the YOLO model
    def reset_ultralytics_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is None or not hasattr(predictor, "trackers"):
            return

        for tracker in predictor.trackers:
            tracker.reset()

        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(predictor.vid_path)

    def get_track_ids(self, result):
        if result.boxes.id is None:
            return [None] * len(result.boxes)

        return result.boxes.id.cpu().int().tolist()

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

    def draw_annotations(self, video_frames, tracks, read_from_stub=False, stub_path=None):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw players, right now player_dict = {track_id: bbox} - track id is passed as will be used to show each different player ID
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0, 0, 255))
                colour = self.brighten_colour(colour)
                frame = self.draw_ellipse(
                    frame, player["bbox"], (colour), track_id)

                # draw the shirt number above the player, if one was confidently read
                number = player.get("number")
                if number is not None:
                    frame = self.draw_number(frame, player["bbox"], number)

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
    
    #  brighten the colours for display
    def brighten_colour(self, colour, factor=1.4):
        colour = np.array(colour, dtype=np.float32)
        colour = colour * factor
        colour = np.clip(colour, 0, 255)
        return tuple(colour.astype(int).tolist())

    # draw the player's shirt number above their bounding box
    def draw_number(self, frame, bbox, number):
        x_center, _ = get_center_of_bbox(bbox)
        y = int(bbox[1]) - 10
        x_text = int(x_center) - 10

        text = str(number)
        # black outline then white fill so it's readable on any kit colour
        cv2.putText(frame, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame
