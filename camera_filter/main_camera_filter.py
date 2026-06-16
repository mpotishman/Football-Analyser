import cv2
import pickle
import os


class MainCameraFilter:
    def __init__(
        self,
        green_threshold=5,
        min_player_count=0,
        debug=False,
        debug_every=25,
        debug_all_flags=False
    ):
        self.green_threshold = green_threshold
        self.min_player_count = min_player_count
        self.debug = debug
        self.debug_every = debug_every
        self.debug_all_flags = debug_all_flags

    def get_main_camera_frame_indexes(self, video_frames, tracks=None):
        keep_flags = []

        for frame_index, frame in enumerate(video_frames):
            player_tracks = None

            if tracks is not None and "players" in tracks:
                player_tracks = tracks["players"][frame_index]

            keep_frame = self.is_main_camera_frame(
                frame,
                frame_index,
                player_tracks
            )
            keep_flags.append(keep_frame)

        if self.debug_all_flags:
            self.print_frame_flags(keep_flags)

        return [
            frame_index for frame_index, keep_frame in enumerate(keep_flags)
            if keep_frame
        ]

    def filter_video_frames(self, video_frames, tracks, read_from_stub=False, stub_path=None):
        main_camera_frame_indexes = None

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                stub_data = pickle.load(f)

            if isinstance(stub_data, tuple):
                main_camera_frame_indexes = stub_data[2]
            else:
                main_camera_frame_indexes = stub_data

        if main_camera_frame_indexes is None:
            main_camera_frame_indexes = self.get_main_camera_frame_indexes(
                video_frames,
                tracks
            )

            # print(
            #     "Frames kept by grass filter: "
            #     f"{len(main_camera_frame_indexes)} / {len(video_frames)}"
            # )

        if not main_camera_frame_indexes:
            raise ValueError("No frames were kept by the grass filter.")

        filtered_video_frames = [
            video_frames[index] for index in main_camera_frame_indexes
        ]
        filtered_tracks = {
            object_type: [
                object_tracks[index] for index in main_camera_frame_indexes
            ]
            for object_type, object_tracks in tracks.items()
        }

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(main_camera_frame_indexes, f)

        return filtered_video_frames, filtered_tracks, main_camera_frame_indexes

    def is_main_camera_frame(self, frame, frame_index=None, player_tracks=None):
        green_pitch_percentage = self.get_green_pitch_percentage(frame)
        player_count = len(player_tracks) if player_tracks is not None else None
        has_enough_green = green_pitch_percentage >= self.green_threshold
        has_enough_players = (
            player_count is None or player_count >= self.min_player_count
        )
        keep_frame = has_enough_green and has_enough_players

        if self.debug and frame_index is not None and frame_index % self.debug_every == 0:
            player_count_text = "unknown"
            if player_count is not None:
                player_count_text = str(player_count)

            print(
                f"Frame {frame_index}: "
                f"green_pitch={green_pitch_percentage:.2f}%, "
                f"players={player_count_text}, "
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

    def print_frame_flags(self, keep_flags, fps=25):
        for frame_index, keep_frame in enumerate(keep_flags):
            timestamp = frame_index / fps
            print(
                f"Frame {frame_index:04d} | "
                f"{timestamp:05.2f}s | "
                f"keep={keep_frame}"
            )
