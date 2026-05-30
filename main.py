# main.py — entry point of the project. Reads the video, runs detection and tracking on it, then saves the output.
# Data flow: raw video frames → tracker assigns IDs to each object → annotated frames saved as output video

from utils import read_video, save_video
from team_assigner import TeamAssigner
from camera_filter import MainCameraFilter
from trackers import Tracker


import cv2
import os


def main():
    video_frames = read_video('input_videos/test_clip_30s.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False,
                                   stub_path='stubs/test_clip_30s_tracks.pkl')
    
    camera_filter = MainCameraFilter(debug=True, debug_every=25)

    main_camera_frame_indexes = camera_filter.get_main_camera_frame_indexes(
        video_frames,
        tracks
    )

    print(f"Main camera frames detected: {len(main_camera_frame_indexes)} / {len(video_frames)}")

    debug_folder = "output_videos/main_camera_debug_frames"
    os.makedirs(debug_folder, exist_ok=True)

    for frame_index in main_camera_frame_indexes[::25]:
        frame = video_frames[frame_index].copy()
        player_tracks = tracks["players"][frame_index]

        for track_id, player in player_tracks.items():
            x1, y1, x2, y2 = [int(value) for value in player["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        output_path = os.path.join(debug_folder, f"frame_{frame_index:06d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"Saved sampled kept frames to {debug_folder}")

    main_camera_frame_index_set = set(main_camera_frame_indexes)
    false_frame_indexes = [
        index for index in range(len(video_frames))
        if index not in main_camera_frame_index_set
    ]

    false_debug_folder = "output_videos/non_game_debug_frames"
    os.makedirs(false_debug_folder, exist_ok=True)

    for frame_index in false_frame_indexes[::25]:
        frame = video_frames[frame_index].copy()
        player_tracks = tracks["players"][frame_index]

        for track_id, player in player_tracks.items():
            x1, y1, x2, y2 = [int(value) for value in player["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        output_path = os.path.join(false_debug_folder, f"frame_{frame_index:06d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"Saved sampled rejected frames to {false_debug_folder}")

    # Assign player teams - just using the first frame so it can get the different colour teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0], tracks['players'][0])

    # assing player to correct team
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_number], track['bbox'], player_id)

            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_colour'] = team_assigner.team_colours[team]

    # draw output
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()
