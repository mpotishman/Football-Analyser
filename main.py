from utils import read_video, save_video
from camera_filter import MainCameraFilter
from gameplay_filter import GameplayFilter
from segments import MainCameraSegmenter
from team_assigner import TeamAssigner
from trackers import Tracker
from numberAssigner import NumberAssigner



MODEL_PATH = "models/best.pt"
VIDEO_PATH = "input_videos/youtube_15_30_16_30.mp4"
STUB_PATH = "stubs/youtube_15_30_16_30_gameplay_tracks.pkl"
OUTPUT_PATH = "output_videos/output_video.avi"

# first get only the video frames
# then keep only gameplay footage using the green pitch filter
# then define the tracks so each object has a tracking id
# then get the start and end frames of main camera segments in the gameplay footage
# assign teams
# get player numbers
# then keep only main camera frames
# draw on image so you can see the players

def main():
    # Define video frames.
    original_video_frames = read_video(VIDEO_PATH)

    # Keep only gameplay footage.
    gameplay_filter = GameplayFilter(
        green_threshold=5,
        debug=True,
        debug_every=25
    )
    video_frames, _ = gameplay_filter.filter_video_frames(
        original_video_frames
    )

    # Define tracks.
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=STUB_PATH,
    )

    # Define camera filter.
    camera_filter = MainCameraFilter(
        green_threshold=5,
        min_player_count=8,
        debug=False,
        debug_every=25
    )

    # Define main camera segments.
    segmenter = MainCameraSegmenter(camera_filter)
    segments = segmenter.get_main_camera_segments(
        video_frames,
        tracks
    )
    print(f"Main camera segments: {segments}")

    # Assign teams.
    team_assigner = TeamAssigner()
    tracks = team_assigner.assign_player_teams(video_frames, tracks)

    # Assign player numbers.
    number_assigner = NumberAssigner(
        confidence_score=0.7
    )

    predicted_numbers = number_assigner.predict_number(
        video_frames,
        segments,
        tracks
    )

    # Keep only main camera frames for output.
    output_video_frames, output_tracks, _ = camera_filter.filter_video_frames(
        video_frames,
        tracks
    )

    # Draw and save output.
    output_video_frames = tracker.draw_annotations(
        output_video_frames,
        output_tracks
    )
    save_video(output_video_frames, OUTPUT_PATH)


if __name__ == "__main__":
    main()
