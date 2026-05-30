import cv2


class MainCameraFilter:
    def __init__(
        self,
        min_players=14,
        max_largest_player_height_ratio=0.30,
        max_median_player_height_ratio=0.18,
        smoothing_window=25,
        min_segment_length=50,
        debug=False,
        debug_every=25
    ):
        self.min_players = min_players
        self.max_largest_player_height_ratio = max_largest_player_height_ratio
        self.max_median_player_height_ratio = max_median_player_height_ratio
        self.smoothing_window = smoothing_window
        self.min_segment_length = min_segment_length
        self.debug = debug
        self.debug_every = debug_every

    def get_main_camera_frame_indexes(self, video_frames, tracks):
        """
        Main public method.

        Takes:
        - video_frames
        - tracks

        Returns:
        - list of frame indexes that should be kept
        """

        # get only the frames where the camera is on the main angle and have the frames be saved as [True, True, false,...] to indicate first 2 frames are main camera
        keep_flags = []
        for index, frame in enumerate(video_frames):
            player_tracks = tracks["players"][index]
            ball_tracks = tracks["ball"][index]
            if self.is_main_camera_frame(frame, player_tracks, ball_tracks, index):
                keep_flags.append(True)
            else:
                keep_flags.append(False)

        return [index for index, keep_frame in enumerate(keep_flags) if keep_frame]

    def is_main_camera_frame(self, frame, player_tracks, ball_tracks, frame_index=None):
        """
        Decides whether one frame looks like gameplay footage.

        Takes:
        - one video frame
        - the player tracks for that frame
        - the ball tracks for that frame

        Returns:
        - True or False
        """

        score = 0
        players_shown = len(player_tracks)
        ball_visible = len(ball_tracks) > 0
        green_pitch_percentage = self.get_green_pitch_percentage(frame)

        if green_pitch_percentage >= 25:
            score += 2

        if ball_visible:
            score += 2

        if players_shown >= 1:
            score += 1

        if players_shown >= 8:
            score += 1

        keep_frame = score >= 3

        if self.debug and frame_index is not None and frame_index % self.debug_every == 0:
            print(
                f"Frame {frame_index}: "
                f"players={players_shown}, "
                f"ball_visible={ball_visible}, "
                f"green_pitch={green_pitch_percentage:.2f}%, "
                f"score={score}, "
                f"keep={keep_frame}"
            )

        return keep_frame

    def get_green_pitch_percentage(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = (35, 40, 40)
        upper_green = (85, 255, 255)

        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = frame.shape[0] * frame.shape[1]

        return (green_pixels / total_pixels) * 100

    def smooth_keep_flags(self, keep_flags):
        """
        Smooths frame-by-frame True/False decisions.

        Takes:
        - list of True/False values

        Returns:
        - smoothed list of True/False values
        """
        pass

    def remove_short_segments(self, keep_flags):
        """
        Removes short kept sections that are probably noise.

        Takes:
        - list of True/False values

        Returns:
        - cleaned list of True/False values
        """
        pass
