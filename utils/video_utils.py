import cv2

# function that appends the frame to the list of frames of the vidoe provided
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# Saves a list of frames as a video file at the given output path
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output_video_frames[0].shape gives (height, width, channels) of the first frame
    # e.g. (1080, 1920, 3) — we pass width then height as VideoWriter expects (width, height)
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()