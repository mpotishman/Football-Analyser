from sklearn.cluster import KMeans
import pickle
import os
import cv2
import numpy as np

class NewTeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}  # player_id : team 1 / 2
        
    def display_tracks(self, tracks):
        for track in tracks["players"]:
            print(f'{track} \n\n')

    # this function walks every frame and prints each player's full track info on one line, in the
    # same style as the tracklet summary - tagging each line with its segment, frame, team and info
    def summarise_tracks(self, tracks, segments):
        for frame_num, player_tracks in enumerate(tracks["players"]):
            if not player_tracks:
                continue
            segment_id = self.segment_for_frame(frame_num, segments)
            for track_id, info in player_tracks.items():
                print(
                    f"Segment {segment_id} | Frame {frame_num} | Track {track_id} | "
                    f"team {info.get('team')} | {info}"
                )

    # helper that returns which segment a frame belongs to, or None if it sits outside every segment
    def segment_for_frame(self, frame_num, segments):
        for segment_id, seg in segments.items():
            if seg["start_frame"] <= frame_num <= seg["end_frame"]:
                return segment_id
        return None



    def assign_player_teams(self, video_frames, tracks, segments, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # get all the colours found within the selected frames in an array format
        all_colours = self.learn_team_colours(video_frames, segments, tracks)
        
        # now cluster these colours into 2 colours
        clustered_colours = self.cluster_team_colours(all_colours)
        
        # call collect_votes, which returns per (segment_id, track_id) tracklet a parallel
        # "frames" and "votes" list (vote is 1 / 2, or None when overlapped / unreadable)
        team_vote_dictionary = self.collect_votes(video_frames, segments, tracks )

        # go through each tracklet, smooth its votes into a stable per-frame team, then write
        # that team (and its colour) back onto every frame the player appears in
        for (segment_id, track_id), verdict in team_vote_dictionary.items():
            final_teams = self.smooth_votes(verdict["votes"])
            self.write_teams_to_tracks(tracks, track_id, verdict["frames"], final_teams)
                
        
        
        
        # for info, teams in team_vote_dictionary.items():
        #     if info[1] == 516:
                
        #         print(f"Segment {info[0]} Track {info[1]} is team {teams.values()} at frame {teams.keys()} \n")
        
        # print("TEAM DICTIONARY: ", team_vote_dictionary)
    
        # for frame_number, frame in enumerate(video_frames):
        #     player_tracks = tracks["players"][frame_number]
        #     for track_id, info in player_tracks.items():
            
                    
        #         bbox = info["bbox"]
        #         team = self.assign_single_player_team(frame, bbox, track_id)
                
        #         if team is None:
        #             tracks["players"][frame_number][track_id]["team_colour"] = (180, 180, 180)
        #         else:
        #             tracks["players"][frame_number][track_id]["team_colour"] = self.team_colours[team]
                    
                
                    
        
        if stub_path is not None:
            self.save_stub(tracks, stub_path)

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
    
    # Builds, per (segment_id, track_id) tracklet, a 'votes' list where each index
    # is the team (1 or 2) assigned to that player in that frame — or None if the
    # player was too overlapped / unreadable in that frame. Aligned 1-to-1 with 'frames'.
    
    # this function takes in the track id and the player tracks - then checks if players bbox is a good frame and if it is it gets the players teams and outputs it ina. dictionary
    # in the format of frame: team. team is NONE if the overlap between boxes is too high  
    def collect_votes(self, video_frames, segments, tracks, occlusion_threshold=0.25):
        tracklets = {}
        for segment_id, seg in segments.items():
            for frame_num in range(seg["start_frame"], seg["end_frame"]):
                player_tracks = tracks["players"][frame_num]
                all_bboxes = [info["bbox"] for info in player_tracks.values()]
                for track_id, info in player_tracks.items():
                    bbox = info["bbox"]
                    vote = None
                    if self.max_overlap(bbox, all_bboxes) <= occlusion_threshold:
                        colour = self.get_player_colour(video_frames[frame_num], bbox)
                        if colour is not None:
                            vote = self.compare_to_team_colours(colour)
                    key = (segment_id, track_id)
                    if key not in tracklets:
                        tracklets[key] = {"frames": [], "votes": []}
                    tracklets[key]["frames"].append(frame_num)
                    tracklets[key]["votes"].append(vote)
        return tracklets

    def max_overlap(self, bbox, all_bboxes):
        best = 0.0
        for other in all_bboxes:
            if other is bbox:
                continue
            overlap = self.overlap_fraction(bbox, other)
            if overlap > best:
                best = overlap
        return best

    def overlap_fraction(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        if area_a == 0:
            return 0.0
        return inter / area_a
    
    # this function takes a tracklet's votes list and smooths it into a stable per-frame team.
    # for each frame it looks at a window of nearby votes and keeps the current team unless the
    # other team clearly takes over that window - this ignores brief overlap flicker but still
    # follows a real, sustained id switch. returns a list of teams (1 / 2 / None) aligned with votes
    def smooth_votes(self, votes, half_window=12, flip_threshold=0.65, min_votes=5):
        final_teams = []
        current_team = None

        for i in range(len(votes)):
            start = max(0, i - half_window)
            end = min(len(votes), i + half_window + 1)

            no_1, no_2 = 0, 0
            for vote in votes[start:end]:
                if vote == 1:
                    no_1 += 1
                elif vote == 2:
                    no_2 += 1

            total = no_1 + no_2

            # only trust the window once it holds enough real (non-None) votes
            if total >= min_votes:
                leader = 1 if no_1 >= no_2 else 2
                leader_share = max(no_1, no_2) / total

                if current_team is None:
                    # the first confident window seeds the team
                    current_team = leader
                elif leader != current_team and leader_share >= flip_threshold:
                    # only flip once the other team clearly dominates the window
                    current_team = leader

            final_teams.append(current_team)

        return final_teams

    # this function writes one decided team onto every frame the player appears in, setting both
    # the team and the team_colour so the drawing step can colour the player correctly
    def write_teams_to_tracks(self, tracks, track_id, frames, final_teams):
        for frame_num, team in zip(frames, final_teams):
            player = tracks["players"][frame_num].get(track_id)
            if player is None:
                continue
            player["team"] = team
            player["team_colour"] = self.colour_for_team(team)

    # this function returns the kit colour for a team, or a neutral grey when the team is unknown
    def colour_for_team(self, team):
        if team is None:
            return (180, 180, 180)
        return self.team_colours[team]


            

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
