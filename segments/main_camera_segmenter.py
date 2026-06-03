class MainCameraSegmenter:
    def __init__(self, camera_filter, min_segment_frames=1):
        self.camera_filter = camera_filter
        self.min_segment_frames = min_segment_frames

    def get_main_camera_segments(self, video_frames, tracks=None):
        segments = {}
        segment_id = 0
        in_main_camera = False
        segment_start_frame = None

        for frame_number, frame in enumerate(video_frames):
            player_tracks = None

            if tracks is not None and "players" in tracks:
                player_tracks = tracks["players"][frame_number]

            is_main_camera = self.camera_filter.is_main_camera_frame(
                frame,
                player_tracks=player_tracks
            )

            if is_main_camera and not in_main_camera:
                segment_start_frame = frame_number
                in_main_camera = True

            elif not is_main_camera and in_main_camera:
                segment_id = self.add_segment(
                    segments,
                    segment_id,
                    segment_start_frame,
                    frame_number - 1
                )

                segment_start_frame = None
                in_main_camera = False

        if in_main_camera:
            self.add_segment(
                segments,
                segment_id,
                segment_start_frame,
                len(video_frames) - 1
            )

        return segments

    def add_segment(self, segments, segment_id, start_frame, end_frame):
        segment_length = end_frame - start_frame + 1

        if segment_length < self.min_segment_frames:
            return segment_id

        segment_id += 1
        segments[segment_id] = {
            "start_frame": start_frame,
            "end_frame": end_frame
        }

        return segment_id
