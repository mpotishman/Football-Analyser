import cv2
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class NumberAssigner:
    def __init__(
        self,
        confidence_score,
        frame_window=30
    ):
        self.confidence_score = confidence_score
        self.frame_window = frame_window

        # PARSeq scene-text recogniser, loaded lazily on first use
        self.recogniser = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # PARSeq expects a 32x128 image normalised to the range [-1, 1]
        self.number_transform = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    
    

    # function to create a big dictionary in the format of tracklets = {
    #     (segment_id, track_id): {
    #         "segment": segment_id,
    #         "track_id": track_id,
    #         "start_frame": first_seen,
    #         "end_frame": last_seen,
    #         "team": team,
    #         "number_predictions": []
    #     }
    # }
    # loop through each segment, then in each frame check if (seg_num, track_id ) is in it, if not make it
    def build_tracklets(self, segments, video_frames, tracks):
        main_dictionary = {}
        for segment_num, seg in segments.items():
            start_frame, end_frame = seg['start_frame'], seg['end_frame']
            for frame_num in range(start_frame, end_frame + 1):
                player_tracks = tracks["players"][frame_num]
                # print("player_tracks: ", player_tracks)
                for track_id, bbox in player_tracks.items():
                    # check the dictionary, if the seg + track id combo is there, update its end frame, otherwise add the start and frame for it
                    if (segment_num, track_id) in main_dictionary:
                        # at least a second of the same track id only
                        # print("CURRENT FRAME: ", frame_num)
                        # print("START FRAME: ", main_dictionary[(segment_num, track_id)]["start_frame"])

                        main_dictionary[(segment_num, track_id)
                                        ]["end_frame"] = frame_num
                        main_dictionary[(segment_num, track_id)]["frames"].append(frame_num)

                    else:
                        main_dictionary[(segment_num, track_id)] = {}
                        self.populate_dictionary(
                            main_dictionary, segment_num, track_id, player_tracks, frame_num)

        
        main_dictionary = self.clean_dictionary(main_dictionary)
        main_dictionary = self.assign_tracklet_teams(main_dictionary, tracks)

        # Now loop through the tracklets, and for each one get the candidate
        for identifier, info in main_dictionary.items():
            candidate_frames = self.select_frames_for_prediction(info)
            info['candidate_frames'] = candidate_frames
            
        # now check these candidate frames and place them into the file
        self.check_candidate(main_dictionary, video_frames, tracks)
        
   

        return main_dictionary, self.make_printable_tracklets(main_dictionary)

     # this function takes in tracklets dictionary, and adds candidate frames to it
    def select_frames_for_prediction(self, track):
        frames = track["frames"]

        if len(frames) <= 10:
            candidate_frames = frames
        else:
            indexes = np.linspace(0, len(frames) - 1, 10, dtype=int)
            candidate_frames = [frames[i] for i in indexes]
        

        return candidate_frames
    
    def make_printable_tracklets(self, dictionary):
        printable = {}

        for identifier, info in dictionary.items():
            printable[identifier] = info.copy()
            printable[identifier].pop("frames", None)

        return printable

    # this function prints each tracklet nicely - one line per (segment_id, track_id) showing its
    # team, frame range and candidate frames, skipping the big per-frame "frames" list
    def summarise_tracklets(self, tracklets):
        for (segment_id, track_id), info in tracklets.items():
            print(
                f"Segment {segment_id} | Track {track_id} | "
                f"team {info.get('team')} | "
                f"frames {info.get('start_frame')}-{info.get('end_frame')} | "
                f"candidates {info.get('candidate_frames')} | "
                f"good candidates {info.get('good_candidates')} | "
                f"number predictions (frame, number, conf) {info.get('number_predictions')}"
            )

    def populate_dictionary(self, dictionary, segment_num, track_id, player_tracks, frame_num):
        dictionary[(segment_num, track_id)]["start_frame"] = frame_num
        dictionary[(segment_num, track_id)]["end_frame"] = frame_num
        dictionary[(segment_num, track_id)]["track_id"] = track_id
        dictionary[(segment_num, track_id)]["team"] = None
        dictionary[(segment_num, track_id)]["segment"] = segment_num
        dictionary[(segment_num, track_id)]["frames"] = [frame_num]

    def clean_dictionary(self, dictionary):
        for identifier, info in list(dictionary.items()):
            if info['end_frame'] - info['start_frame'] < self.frame_window:
                dictionary.pop(identifier)
                
        return dictionary

    def assign_tracklet_teams(self, dictionary, tracks):
        for identifier, info in dictionary.items():
            track_id = info["track_id"]
            team_counts = {}

            for frame in info["frames"]:
                player = tracks["players"][frame].get(track_id)
                if player is None:
                    continue

                team = player.get("team")
                if team is None:
                    continue

                team_counts[team] = team_counts.get(team, 0) + 1

            if team_counts:
                info["team"] = max(team_counts, key=team_counts.get)

        return dictionary
    
    # now using the cnadidate frames in each tracklet, check if they are GOOD crops using gdef good cropping
    # then using the final frames - send it to the uncertainty
    def check_candidate(self, dictionary, video_frames, tracks):
        for identifier, info in list(dictionary.items()):
            segment_id, track_id = identifier
            candidates = info['candidate_frames']
            good_candidates = []
            number_predictions = []
            for frame in candidates:
                frame_image = video_frames[frame]
                player_bbox = tracks["players"][frame][track_id]["bbox"]
                x1, y1, x2, y2 = player_bbox

                h, w = frame_image.shape[:2]

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                height = y2 - y1
                width = x2 - x1

                player_crop = frame_image[y1:y2, x1:x2]
                
                if self.good_cropping(player_crop, height, width):
                    # use this crop - later if need to be placed in folder use code bELOW
                    # crop_folder = f"crops/segment_{segment_id}/track_{track_id}"
                    # os.makedirs(crop_folder, exist_ok=True)

                    # crop_path = (
                    #     f"{crop_folder}/"
                    #     f"segment_{segment_id}_track_{track_id}_frame_{frame}.jpg"
                    # )

                    # cv2.imwrite(crop_path, player_crop)
                    
                    # now if its a good crop, run the kit prediction on it, not really needed but in case needed later
                    good_candidates.append(frame)
                    
                    # now call the prediction in the form of number_predictions = [
                        # {"frame": 734, "number": "56", "confidence": 0.83}]
                        
                    player_kit_prediction, confidence = self.predict_number(player_crop)
                    if player_kit_prediction is not None:
                        number_predictions.append((frame, player_kit_prediction, confidence))

            info['good_candidates'] = good_candidates
            info['number_predictions'] = number_predictions

        return dictionary
                    
                    
                                
    
    # now use the uncertainty model on the candidate frames
    def good_cropping(self, player_crop, height, width):
        if height < 80 or width < 30:
            return False

        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 50:
            # blurry
            return False

        return True

    # loads the PARSeq scene-text recogniser once and caches it on the instance
    def load_recogniser(self):
        if self.recogniser is None:
            self.recogniser = torch.hub.load(
                'baudm/parseq', 'parseq', pretrained=True, trust_repo=True
            ).eval().to(self.device)
        return self.recogniser
    
        
        
    

    # runs PARSeq on a crop and returns the raw recognised (text, confidence) with NO filtering.
    # confidence is the whole-string probability (the uncertainty). useful for debugging what the
    # model actually read before the digit check throws it away
    def read_text(self, crop):
        parseq = self.load_recogniser()

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.number_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = parseq(tensor)
            pred = logits.softmax(-1)
            labels, confidences = parseq.tokenizer.decode(pred)

        conf = confidences[0]
        score = float(conf.prod()) if len(conf) else 0.0

        return labels[0], score

    # crops the upper-back region of a full player crop, where the shirt number sits, so the
    # recogniser sees mostly the digits instead of the whole body. this is a rough fixed-fraction
    # heuristic - pose estimation would locate the number far more reliably
    def get_number_crop(self, player_crop):
        h, w = player_crop.shape[:2]
        y1 = int(h * 0.20)
        y2 = int(h * 0.50)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        return player_crop[y1:y2, x1:x2]

    # reads the shirt number from one player crop and returns (number_string, confidence), or
    # (None, 0.0) if the read is not a valid jersey number (non-digit or > 99)
    def predict_number(self, player_crop):
        number_crop = self.get_number_crop(player_crop)
        text, score = self.read_text(number_crop)

        if not text.isdigit() or int(text) > 99:
            return None, 0.0

        return text, score
