import cv2


class GameplayFilter:
    def __init__(self, green_threshold=5, debug=False, debug_every=25):
        self.green_threshold = green_threshold
        self.debug = debug
        self.debug_every = debug_every

    def get_gameplay_frame_indexes(self, video_frames):
        gameplay_frame_indexes = []

        for frame_index, frame in enumerate(video_frames):
            keep_frame = self.is_gameplay_frame(frame, frame_index)

            if keep_frame:
                gameplay_frame_indexes.append(frame_index)

        return gameplay_frame_indexes

    def filter_video_frames(self, video_frames):
        gameplay_frame_indexes = self.get_gameplay_frame_indexes(video_frames)

        print(
            "Frames kept by gameplay filter: "
            f"{len(gameplay_frame_indexes)} / {len(video_frames)}"
        )

        if not gameplay_frame_indexes:
            raise ValueError("No frames were kept by the gameplay filter.")

        filtered_video_frames = [
            video_frames[index] for index in gameplay_frame_indexes
        ]

        return filtered_video_frames, gameplay_frame_indexes

    def is_gameplay_frame(self, frame, frame_index=None):
        green_pitch_percentage = self.get_green_pitch_percentage(frame)
        keep_frame = green_pitch_percentage >= self.green_threshold

        if self.debug and frame_index is not None and frame_index % self.debug_every == 0:
            print(
                f"Frame {frame_index}: "
                f"green_pitch={green_pitch_percentage:.2f}%, "
                f"gameplay={keep_frame}"
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
