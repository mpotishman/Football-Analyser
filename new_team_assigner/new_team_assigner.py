from sklearn.cluster import KMeans
import pickle
import os
import cv2
import numpy as np


class NewTeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}  # player_id : team 1 / 2

    def assign_player_teams(self, video_frames, tracks, segments, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # get all the colours found within the selected frames in an array format
        all_colours = self.learn_team_colours(video_frames, segments, tracks)
        
        # now cluster these colours into 2 colours
        clustered_colours = self.cluster_team_colours(all_colours)
    
        for frame_number, frame in enumerate(video_frames):
            player_tracks = tracks["players"][frame_number]
            for track_id, info in player_tracks.items():
                bbox = info["bbox"]
                team = self.assign_single_player_team(frame, bbox, track_id)
                tracks["players"][frame_number][track_id]["team"] = team
                if team is None:
                    tracks["players"][frame_number][track_id]["team_colour"] = (180, 180, 180)
                else:
                    tracks["players"][frame_number][track_id]["team_colour"] = self.team_colours[team]
                    
        
        if stub_path is not None:
            self.save_stub(tracks, stub_path)

        return tracks
                
        return tracks

    def learn_team_colours(self, video_frames, segments, tracks):
        # now learn the teams colours by looking at the frames
        # output => self.team_colours = {
        #     1: red_team_colour,
        #     2: yellow_team_colour
        # }
        # get the good frames, crop the players out of it and get the player colour
        # add that colour to player_colours, then use kmeans to cluser player_colours into 2 colours

        # get the good frames
        good_frames = self.select_learning_frames(segments, tracks)
        
        # go through each frame, then get the tracks from that frame and go through each bbox, extract the kit colour, and add it to the colours array to later be clustered into 2 colours
        all_colours = []
        for frame in good_frames:
            chosen_frame_tracks = tracks["players"][frame]

            for track_id, info in chosen_frame_tracks.items():
                
                kit_colour = self.get_player_colour(video_frames[frame], info["bbox"])
                if kit_colour is not None:
                    all_colours.append(kit_colour)
                
    
        return all_colours

    def select_learning_frames(self, segments, tracks):
        # this function checks the frame and outputs a list of numbers that are frame numbers of those that are good
        frames_for_learning = []
        for segment_id, seg in segments.items():
            for frame in range(seg['start_frame'], seg['end_frame'], 10):
                if self.is_good_learning_frame(tracks["players"][frame]):
                    frames_for_learning.append(frame)
                    
        return frames_for_learning

    def is_good_learning_frame(self, track):
        # now check if its good by seeing how many players there are
        if len(track) < 8:
            return False
        return True

    # function to now get an individual players colour
    def get_player_colour(self, frame, bbox):
        player_crop = self.get_player_crop(frame, bbox)

        if player_crop is None:
            return None

        if not self.is_good_crop(player_crop):
            return None

        pixels = self.extract_kit_pixels(player_crop)

        if len(pixels) == 0:
            return None

        player_colour = np.median(pixels, axis=0)

        return player_colour

    # get the cropped image of the frame
    def get_player_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        player_crop = frame[y1:y2, x1:x2]

        h, w = player_crop.shape[:2]

        torso_y1 = int(h * 0.15)
        torso_y2 = int(h * 0.55)
        torso_x1 = int(w * 0.20)
        torso_x2 = int(w * 0.80)

        torso_crop = player_crop[torso_y1:torso_y2, torso_x1:torso_x2]

        return torso_crop
        
    # see if its a good cropping
    def is_good_crop(self, player_crop):
        height, width = player_crop.shape[:2]
        if height < 30 or width < 15:
            return False

        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 50:
            # blurry
            return False

        return True

    # get only the pixels from the kit
    def extract_kit_pixels(self, player_crop):
        if player_crop is None or player_crop.size == 0:
            return np.array([])

        hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Remove green pitch
        green_pixels = (
            (hue >= 35) &
            (hue <= 85) &
            (saturation > 40) &
            (value > 40)
        )

        # Remove dark shadow pixels
        dark_pixels = value < 50

        # Remove grey-ish pixels
        grey_pixels = saturation < 60

        # Remove bright white-ish pixels
        white_pixels = (saturation < 80) & (value > 180)

        kit_mask = ~green_pixels & ~dark_pixels & ~grey_pixels & ~white_pixels

        kit_pixels = player_crop[kit_mask]

        return kit_pixels
    
    def cluster_team_colours(self, all_colours):
        valid_colours = []

        for colour in all_colours:
            if isinstance(colour, np.ndarray) and len(colour) == 3:
                valid_colours.append(colour)

        valid_colours = np.array(valid_colours)

        if len(valid_colours) < 2:
            raise ValueError("Not enough valid kit colours to cluster teams.")
        
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(valid_colours)

        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]
        
        return self.team_colours

    def assign_single_player_team(self, frame, bbox, track_id):
        current_colour = self.get_player_colour(frame, bbox)

        if current_colour is None:
            if self.should_use_cached_team(track_id, current_colour):
                return self.player_team_dict[track_id]

            return None

        team = self.compare_to_team_colours(current_colour)

        self.player_team_dict[track_id] = team

        return team

    def compare_to_team_colours(self, player_colour):
        distance_to_team_1 = np.linalg.norm(player_colour - self.team_colours[1])
        distance_to_team_2 = np.linalg.norm(player_colour - self.team_colours[2])

        if distance_to_team_1 < distance_to_team_2:
            return 1

        return 2

    # this function checks to see if it should recaclulate a players colour, if the track id not in there, or player colour is NOT none
    def should_use_cached_team(self, track_id, player_colour=None):
            if track_id not in self.player_team_dict:
                return False
            
            if player_colour is not None:
                return False
            
            return True

    def load_stub(self, stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)

        return tracks


    def save_stub(self, tracks, stub_path):
        with open(stub_path, "wb") as f:
            pickle.dump(tracks, f)
