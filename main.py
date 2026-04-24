# main.py — entry point of the project. Reads the video, runs detection and tracking on it, then saves the output.
# Data flow: raw video frames → tracker assigns IDs to each object → annotated frames saved as output video

from utils import read_video, save_video
from team_assigner import TeamAssigner
from trackers import Tracker
import cv2


def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Assign player teams - just using the first frame so it can get the different colour teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0], tracks['players'][0])
    
    # assing player to correct team
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_number], track['bbox'], player_id)
            
            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_colour'] = team_assigner.team_colours[team]
            
    
   
    # draw output
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_video.avi')
    

if __name__ == '__main__':
    main()
