# main.py — entry point of the project. Reads the video, runs detection and tracking on it, then saves the output.
# Data flow: raw video frames → tracker assigns IDs to each object → annotated frames saved as output video

from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # draw output
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_video.avi')
    
    # test


if __name__ == '__main__':
    main()
