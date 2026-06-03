import cv2
import os


class NumberAssigner:
    def __init__(
        self,
        confidence_score
    ): 
        self.confidence_score = confidence_score
        
        
    # go through each segment to get the frames for it
    # choose every nth frame
    # for that frame go through the tracks and select them to be cropped and tested for which number they are wearing
    def predict_number(self, video_frames, segments, tracks=None):
        predicted_numbers = {}
        n = 5
        max_crops_per_player = 6
        os.makedirs("crops", exist_ok=True)

        for segment_id, seg in segments.items():
            num_crops = {}

            for frame_number in range(seg["start_frame"], seg["end_frame"] + 1, n):
                frame = video_frames[frame_number]
                player_tracks = tracks["players"][frame_number]

                for player_id, track in player_tracks.items():
                    if player_id not in num_crops:
                        num_crops[player_id] = 0

                    if num_crops[player_id] >= max_crops_per_player:
                        continue

                    bbox = track["bbox"]

                    x1, y1, x2, y2 = bbox
                    h, w = frame.shape[:2]

                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    
                    height = y2 - y1
                    width = x2-x1

                    if height <= 0 or width <= 0:
                        continue

                    player_crop = frame[y1:y2, x1:x2]
                    
                    if self.good_cropping(player_crop, height, width):
                        # then save it and use it later to get the actual number, pass in the player id as well
                        crop_folder = f"crops/segment_{segment_id}/track_{player_id}"
                        os.makedirs(crop_folder, exist_ok=True)

                        cv2.imwrite(
                            f"{crop_folder}/frame_{frame_number}.jpg",
                            player_crop
                        )
                        
                        num_crops[player_id] += 1
                        
                        # run number model here
        
        return predicted_numbers

    def good_cropping(self, player_crop, height, width):
        if height < 80 or width < 30:
            return False
        
        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < 50:
            # blurry
            return False
        
        return True
