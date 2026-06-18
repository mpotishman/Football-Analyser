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
        frame_window=30,
        min_support=2,   # a number must be read on at least this many frames to be accepted
        legibility_threshold=0.5,  # crop must score above this to be considered legible
        max_candidate_frames=30,
        min_tracklet_frames=8,
        max_candidate_overlap=0.20,
    ):
        self.confidence_score = confidence_score
        self.frame_window = frame_window
        self.min_support = min_support
        self.legibility_threshold = legibility_threshold
        self.max_candidate_frames = max_candidate_frames
        self.min_tracklet_frames = min_tracklet_frames
        self.max_candidate_overlap = max_candidate_overlap

        # PARSeq scene-text recogniser, loaded lazily on first use
        self.recogniser = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # legibility classifier (is a number visible at all?), loaded lazily on first use
        self.legibility = None

        # PARSeq expects a 32x128 image normalised to the range [-1, 1]
        self.number_transform = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

        # ResNet34 legibility classifier expects a 224x224 ImageNet-normalised image
        self.legibility_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    
    

    # function to create a big dictionary in the format of tracklets = {
    #     (segment_id, track_id, split_id): {
    #         "segment": segment_id,
    #         "track_id": track_id,
    #         "split_id": split_id,
    #         "start_frame": first_seen,
    #         "end_frame": last_seen,
    #         "team": team,
    #         "number_predictions": []
    #     }
    # }
    # each raw YOLO/BoT-SORT track_id can become multiple split_id runs if it jumps players.
    def build_tracklets(self, segments, video_frames, tracks):
        main_dictionary = {}
        active_tracklets = {}
        next_split_ids = {}

        for segment_num, seg in segments.items():
            start_frame, end_frame = seg['start_frame'], seg['end_frame']
            for frame_num in range(start_frame, end_frame + 1):
                player_tracks = tracks["players"][frame_num]
                # print("player_tracks: ", player_tracks)
                for track_id, player in player_tracks.items():
                    base_key = (segment_num, track_id)
                    active_key = active_tracklets.get(base_key)

                    if active_key is None or self.should_start_new_tracklet(
                        main_dictionary[active_key],
                        frame_num,
                        player,
                        tracks
                    ):
                        split_id = next_split_ids.get(base_key, 0)
                        identifier = (segment_num, track_id, split_id)
                        next_split_ids[base_key] = split_id + 1
                        active_tracklets[base_key] = identifier
                        main_dictionary[identifier] = {}
                        self.populate_dictionary(
                            main_dictionary,
                            segment_num,
                            track_id,
                            split_id,
                            player,
                            frame_num
                        )
                    else:
                        self.update_dictionary(
                            main_dictionary,
                            active_key,
                            player,
                            frame_num
                        )

        
        main_dictionary = self.clean_dictionary(main_dictionary)
        main_dictionary = self.assign_tracklet_teams(main_dictionary, tracks)

        # Now loop through the tracklets, and for each one get the candidate
        for identifier, info in main_dictionary.items():
            candidate_frames = self.select_frames_for_prediction(info)
            info['candidate_frames'] = candidate_frames
            
        # now check these candidate frames and place them into the file
        self.check_candidate(main_dictionary, video_frames, tracks)

        # collapse each tracklet's per-frame reads into one final number (the most popular)
        self.assign_final_predictions(main_dictionary)

        # if two same-team players overlap in time and receive the same number, keep the stronger read
        self.resolve_number_conflicts(main_dictionary)

        # write each tracklet's number predictions + final number back onto tracks, per frame
        self.match_predictions_to_tracks(main_dictionary, tracks)

        return main_dictionary, self.make_printable_tracklets(main_dictionary)

     # this function takes in tracklets dictionary, and adds candidate frames to it
    def select_frames_for_prediction(self, track):
        frames = track["frames"]

        if len(frames) <= self.max_candidate_frames:
            candidate_frames = frames
        else:
            indexes = np.linspace(0, len(frames) - 1, self.max_candidate_frames, dtype=int)
            candidate_frames = [frames[i] for i in indexes]
        

        return candidate_frames
    
    def make_printable_tracklets(self, dictionary):
        printable = {}

        for identifier, info in dictionary.items():
            printable[identifier] = info.copy()
            printable[identifier].pop("frames", None)

        return printable

    # this function looks at a tracklet's number_predictions and returns the single most popular
    # number (by how many frames read it), or None if there are no predictions. ties are broken
    # by the higher total confidence.
    def get_final_prediction(self, number_predictions):
        best, _support, _confidence = self.get_final_prediction_with_stats(number_predictions)
        return best

    def get_final_prediction_with_stats(self, number_predictions):
        if not number_predictions:
            return None, 0, 0.0

        counts = {}
        conf_total = {}
        for prediction in number_predictions:
            if len(prediction) >= 3:
                _frame, number, conf = prediction[:3]
            else:
                continue
            counts[number] = counts.get(number, 0) + 1
            conf_total[number] = conf_total.get(number, 0.0) + conf

        if not counts:
            return None, 0, 0.0

        best = max(counts, key=lambda n: (counts[n], conf_total[n]))

        # require the number to be read on enough frames, else it's likely a hallucination
        if counts[best] < self.min_support:
            return None, counts[best], conf_total[best]

        return best, counts[best], conf_total[best]

    def assign_final_predictions(self, dictionary):
        for identifier, info in dictionary.items():
            best, support, confidence = self.get_final_prediction_with_stats(
                info.get('number_predictions', [])
            )
            info['final_prediction'] = best
            info['final_support'] = support
            info['final_confidence'] = confidence

        return dictionary

    def resolve_number_conflicts(self, dictionary):
        grouped_tracklets = {}
        for identifier, info in dictionary.items():
            number = info.get("final_prediction")
            team = info.get("team")
            if number is None or team is None:
                continue

            key = (info.get("segment"), team, number)
            if key not in grouped_tracklets:
                grouped_tracklets[key] = []
            grouped_tracklets[key].append((identifier, info))

        for _group_key, tracklets in grouped_tracklets.items():
            for i in range(len(tracklets)):
                first_identifier, first_info = tracklets[i]
                if first_info.get("final_prediction") is None:
                    continue

                for j in range(i + 1, len(tracklets)):
                    second_identifier, second_info = tracklets[j]
                    if second_info.get("final_prediction") is None:
                        continue

                    if not self.tracklets_overlap(first_info, second_info):
                        continue

                    loser_identifier, loser_info = self.weaker_tracklet(
                        first_identifier,
                        first_info,
                        second_identifier,
                        second_info
                    )
                    loser_info["conflict_reason"] = (
                        "same team number overlaps another stronger tracklet"
                    )
                    loser_info["final_prediction"] = None

        return dictionary

    def tracklets_overlap(self, first_info, second_info, min_overlap_frames=10):
        first_frames = set(first_info.get("frames", []))
        second_frames = set(second_info.get("frames", []))
        return len(first_frames.intersection(second_frames)) >= min_overlap_frames

    def weaker_tracklet(self, first_identifier, first_info, second_identifier, second_info):
        first_score = self.tracklet_prediction_score(first_info)
        second_score = self.tracklet_prediction_score(second_info)

        if first_score >= second_score:
            return second_identifier, second_info
        return first_identifier, first_info

    def tracklet_prediction_score(self, info):
        return (
            info.get("final_support", 0),
            info.get("final_confidence", 0.0),
            len(info.get("good_candidates", [])),
            len(info.get("frames", []))
        )

    # writes each tracklet's final number back onto tracks, so every frame the player appears in
    # carries its shirt number plus the start/end frame of the run that number applies to
    def match_predictions_to_tracks(self, dictionary, tracks):
        for identifier, info in dictionary.items():
            track_id = info["track_id"]
            number = info.get("final_prediction")
            start_frame = info.get("start_frame")
            end_frame = info.get("end_frame")
            split_id = info.get("split_id")
            for frame in info["frames"]:
                player = tracks["players"][frame].get(track_id)
                if player is not None:
                    player["number"] = number
                    player["number_start_frame"] = start_frame
                    player["number_end_frame"] = end_frame
                    player["number_split_id"] = split_id

        return tracks

    # this function prints each tracklet nicely - one line per (segment_id, track_id) showing its
    # team, frame range and candidate frames, skipping the big per-frame "frames" list
    def summarise_tracklets(self, tracklets):
        for identifier, info in tracklets.items():
            segment_id, track_id = identifier[:2]
            split_id = info.get("split_id", 0)
            print(
                f"Segment {segment_id} | Track {track_id} | "
                f"split {split_id} | "
                f"team {info.get('team')} | "
                f"FINAL PREDICTION {info.get('final_prediction')} | "
                f"support {info.get('final_support')} | "
                f"frames {info.get('start_frame')}-{info.get('end_frame')} | "
                f"candidates {info.get('candidate_frames')} | "
                f"good candidates {info.get('good_candidates')} | "
                f"number predictions (frame, number, conf) {info.get('number_predictions')}"
            )

    def populate_dictionary(self, dictionary, segment_num, track_id, split_id, player, frame_num):
        identifier = (segment_num, track_id, split_id)
        bbox = player.get("bbox")
        team = player.get("team")

        dictionary[identifier]["start_frame"] = frame_num
        dictionary[identifier]["end_frame"] = frame_num
        dictionary[identifier]["last_frame"] = frame_num
        dictionary[identifier]["track_id"] = track_id
        dictionary[identifier]["split_id"] = split_id
        dictionary[identifier]["team"] = None
        dictionary[identifier]["last_known_team"] = team
        dictionary[identifier]["segment"] = segment_num
        dictionary[identifier]["frames"] = [frame_num]
        dictionary[identifier]["last_bbox"] = bbox

    def update_dictionary(self, dictionary, identifier, player, frame_num):
        bbox = player.get("bbox")
        team = player.get("team")

        dictionary[identifier]["end_frame"] = frame_num
        dictionary[identifier]["last_frame"] = frame_num
        dictionary[identifier]["frames"].append(frame_num)
        dictionary[identifier]["last_bbox"] = bbox
        if team is not None:
            dictionary[identifier]["last_known_team"] = team

    def clean_dictionary(self, dictionary):
        for identifier, info in list(dictionary.items()):
            frame_span = info['end_frame'] - info['start_frame']
            frame_count = len(info.get("frames", []))
            if frame_span < self.frame_window or frame_count < self.min_tracklet_frames:
                dictionary.pop(identifier)
                
        return dictionary

    def should_start_new_tracklet(self, tracklet, frame_num, player, tracks):
        if self.has_frame_gap(tracklet, frame_num):
            return True

        if self.has_team_switch(tracklet, player):
            return True

        if self.has_bbox_jump(tracklet, player):
            return True

        return False

    def has_frame_gap(self, tracklet, frame_num, max_gap=5):
        last_frame = tracklet.get("last_frame")
        if last_frame is None:
            return False
        return frame_num - last_frame > max_gap

    def has_team_switch(self, tracklet, player):
        previous_team = tracklet.get("last_known_team")
        current_team = player.get("team")

        if previous_team is None or current_team is None:
            return False

        return current_team != previous_team

    def has_bbox_jump(self, tracklet, player):
        previous_bbox = tracklet.get("last_bbox")
        current_bbox = player.get("bbox")

        if previous_bbox is None or current_bbox is None:
            return False

        previous_center = self.get_bbox_center(previous_bbox)
        current_center = self.get_bbox_center(current_bbox)
        center_distance = np.linalg.norm(np.array(previous_center) - np.array(current_center))

        previous_height = max(1.0, previous_bbox[3] - previous_bbox[1])
        current_height = max(1.0, current_bbox[3] - current_bbox[1])
        previous_area = max(1.0, self.get_bbox_area(previous_bbox))
        current_area = max(1.0, self.get_bbox_area(current_bbox))
        area_ratio = current_area / previous_area
        iou = self.bbox_iou(previous_bbox, current_bbox)

        if center_distance > max(120.0, previous_height * 2.0) and iou < 0.02:
            return True

        if area_ratio > 3.0 or area_ratio < 0.33:
            height_ratio = current_height / previous_height
            if height_ratio > 2.0 or height_ratio < 0.5:
                return True

        return False

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def get_bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def bbox_iou(self, first_bbox, second_bbox):
        ax1, ay1, ax2, ay2 = first_bbox
        bx1, by1, bx2, by2 = second_bbox

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h

        area_a = self.get_bbox_area(first_bbox)
        area_b = self.get_bbox_area(second_bbox)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / union

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
        items = list(dictionary.items())
        total = len(items)
        for i, (identifier, info) in enumerate(items, start=1):
            segment_id, track_id = identifier[:2]
            split_id = info.get("split_id", 0)
            print(
                f"Reading numbers: tracklet {i}/{total} "
                f"(segment {segment_id}, track {track_id}, split {split_id})"
            )

            # skip number guessing for tracklets with no assigned team - they're unreliable
            if info.get('team') is None:
                info['good_candidates'] = []
                info['number_predictions'] = []
                info['candidate_details'] = []
                continue

            candidates = info['candidate_frames']
            good_candidates = []
            number_predictions = []
            candidate_details = []
            for frame in candidates:
                frame_image = video_frames[frame]
                player = tracks["players"][frame].get(track_id)
                if player is None:
                    candidate_details.append({
                        "frame": frame,
                        "used": False,
                        "reason": "track missing"
                    })
                    continue

                player_bbox = player["bbox"]
                all_bboxes = [other["bbox"] for other in tracks["players"][frame].values()]
                overlap = self.max_overlap(player_bbox, all_bboxes)

                if overlap > self.max_candidate_overlap:
                    candidate_details.append({
                        "frame": frame,
                        "used": False,
                        "reason": "overlap",
                        "overlap": overlap
                    })
                    continue

                player_crop, height, width = self.get_player_crop_from_bbox(frame_image, player_bbox)
                if not self.good_cropping(player_crop, height, width):
                    candidate_details.append({
                        "frame": frame,
                        "used": False,
                        "reason": "bad crop",
                        "overlap": overlap,
                        "crop_size": (height, width)
                    })
                    continue

                # use this crop - later if need to be placed in folder use code bELOW
                # crop_folder = f"crops/segment_{segment_id}/track_{track_id}/split_{split_id}"
                # os.makedirs(crop_folder, exist_ok=True)

                # crop_path = (
                #     f"{crop_folder}/"
                #     f"segment_{segment_id}_track_{track_id}_split_{split_id}_frame_{frame}.jpg"
                # )

                # cv2.imwrite(crop_path, player_crop)

                good_candidates.append(frame)
                legibility_score = self.get_legibility_score(player_crop)

                if legibility_score < self.legibility_threshold:
                    candidate_details.append({
                        "frame": frame,
                        "used": False,
                        "reason": "not legible",
                        "overlap": overlap,
                        "legibility": legibility_score,
                        "crop_size": (height, width)
                    })
                    continue

                # now call the prediction in the form of number_predictions = [
                    # {"frame": 734, "number": "56", "confidence": 0.83}]

                player_kit_prediction, confidence, crop_name, raw_text = self.predict_number_with_details(
                    player_crop
                )
                candidate_details.append({
                    "frame": frame,
                    "used": player_kit_prediction is not None and confidence >= self.confidence_score,
                    "reason": "read",
                    "overlap": overlap,
                    "legibility": legibility_score,
                    "crop_size": (height, width),
                    "crop_name": crop_name,
                    "raw_text": raw_text,
                    "prediction": player_kit_prediction,
                    "confidence": confidence
                })

                if player_kit_prediction is not None and confidence >= self.confidence_score:
                    number_predictions.append((frame, player_kit_prediction, confidence, crop_name))

            info['good_candidates'] = good_candidates
            info['number_predictions'] = number_predictions
            info['candidate_details'] = candidate_details

        return dictionary
                    
                    
                                
    
    # now use the uncertainty model on the candidate frames
    def good_cropping(self, player_crop, height, width):
        if player_crop is None or player_crop.size == 0:
            return False

        if height < 80 or width < 30:
            return False

        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 50:
            # blurry
            return False

        return True

    def get_player_crop_from_bbox(self, frame_image, player_bbox):
        x1, y1, x2, y2 = player_bbox
        h, w = frame_image.shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        height = y2 - y1
        width = x2 - x1

        if height <= 0 or width <= 0:
            return None, height, width

        player_crop = frame_image[y1:y2, x1:x2]
        return player_crop, height, width

    def max_overlap(self, bbox, all_bboxes):
        best = 0.0
        for other in all_bboxes:
            if other is bbox:
                continue
            overlap = self.overlap_fraction(bbox, other)
            if overlap > best:
                best = overlap
        return best

    def overlap_fraction(self, first_bbox, second_bbox):
        ax1, ay1, ax2, ay2 = first_bbox
        bx1, by1, bx2, by2 = second_bbox
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        if area_a <= 0:
            return 0.0
        return inter / area_a

    # loads the ResNet34 legibility classifier once and caches it on the instance. it answers
    # "is a jersey number actually visible in this crop?" so we can skip PARSeq on crops that
    # clearly have no number (player facing away/sideways, occluded, etc.)
    def load_legibility(self):
        if self.legibility is None:
            import torchvision
            model = torchvision.models.resnet34(weights=None)
            model.fc = torch.nn.Linear(512, 1)
            state = torch.load("models/legibility_resnet34.pth", map_location=self.device, weights_only=False)
            # the checkpoint prefixes every key with "model_ft." - strip it to match a bare resnet34
            state = {k.replace("model_ft.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
            self.legibility = model.eval().to(self.device)
        return self.legibility

    # returns True if the crop looks like it contains a readable number, False otherwise
    def is_legible(self, player_crop):
        return self.get_legibility_score(player_crop) >= self.legibility_threshold

    def get_legibility_score(self, player_crop):
        model = self.load_legibility()
        rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        tensor = self.legibility_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = torch.sigmoid(model(tensor)).item()
        return score

    # loads the PARSeq scene-text recogniser once and caches it on the instance
    def load_recogniser(self):
        if self.recogniser is None:
            import sys
            # reuse the parseq source torch.hub already downloaded, so we can load a local checkpoint
            parseq_repo = os.path.join(torch.hub.get_dir(), "baudm_parseq_main")
            if parseq_repo not in sys.path:
                sys.path.insert(0, parseq_repo)
            from strhub.models.utils import load_from_checkpoint # type: ignore

            # Torch 2.6 defaults torch.load to weights_only=True, which rejects this trusted
            # Lightning checkpoint. Force weights_only=False just for this load, then restore.
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(
                *args, **{**kwargs, "weights_only": False}
            )
            try:
                self.recogniser = load_from_checkpoint(
                    "models/parseq_jersey_remap.ckpt"
                ).eval().to(self.device)
            finally:
                torch.load = original_torch_load
        return self.recogniser

        
        
    

    # runs PARSeq on a crop and returns the raw recognised (text, confidence) with NO filtering.
    # confidence is the whole-string probability (the uncertainty). useful for debugging what the
    # model actually read before the digit check throws it away
    def read_text(self, crop):
        if crop is None or crop.size == 0:
            return "", 0.0

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
        return self.get_number_crop_options(player_crop)["upper_back"]

    def get_number_crop_options(self, player_crop):
        h, w = player_crop.shape[:2]
        crop_specs = {
            "upper_back": (0.20, 0.50, 0.15, 0.85),
            "wide_upper_back": (0.18, 0.55, 0.05, 0.95),
            "middle_back": (0.28, 0.62, 0.12, 0.88),
            "high_torso": (0.12, 0.45, 0.10, 0.90),
        }

        crops = {}
        for name, (y1_ratio, y2_ratio, x1_ratio, x2_ratio) in crop_specs.items():
            y1 = int(h * y1_ratio)
            y2 = int(h * y2_ratio)
            x1 = int(w * x1_ratio)
            x2 = int(w * x2_ratio)
            crops[name] = player_crop[y1:y2, x1:x2]

        return crops

    # reads the shirt number from one player crop and returns (number_string, confidence), or
    # (None, 0.0) if the read is not a valid jersey number (non-digit or > 99)
    def predict_number(self, player_crop):
        number, score, _crop_name, _raw_text = self.predict_number_with_details(player_crop)
        return number, score

    def predict_number_with_details(self, player_crop):
        best_number = None
        best_valid_score = 0.0
        best_raw_score = 0.0
        best_crop_name = None
        best_raw_text = ""

        for crop_name, number_crop in self.get_number_crop_options(player_crop).items():
            text, score = self.read_text(number_crop)
            if score > best_raw_score:
                best_raw_score = score
                best_raw_text = text
                best_crop_name = crop_name

            if not text.isdigit() or int(text) > 99:
                continue

            if score > best_valid_score:
                best_number = text
                best_valid_score = score
                best_crop_name = crop_name
                best_raw_text = text

        if best_number is None:
            return None, best_raw_score, best_crop_name, best_raw_text

        return best_number, best_valid_score, best_crop_name, best_raw_text
