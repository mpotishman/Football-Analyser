import cv2
from collections import OrderedDict


class LazyVideoFrames:
    """Reads frames from a video on demand instead of loading them all into RAM.
    Acts like a list: supports len(), indexing, slicing and iteration."""

    def __init__(self, video_path, cache_size=64):
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        self._count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._next_index = 0  # avoids a seek when reading sequentially

    def __len__(self):
        return self._count

    def _read_index(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if index != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_index = index
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(f"Could not read frame {index} from {self.video_path}")
        self._next_index += 1
        self._cache[index] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._read_index(i) for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        return self._read_index(index)

    def __iter__(self):
        for i in range(self._count):
            yield self._read_index(i)

    def release(self):
        self._cap.release()


class FrameSubset:
    """A lazy, re-indexed view of a subset of frames: view[j] == base[indexes[j]].
    Replaces [base[i] for i in indexes] without copying frames into a new list."""

    def __init__(self, base, indexes):
        self.base = base
        self.indexes = list(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.base[self.indexes[i]] for i in range(*index.indices(len(self.indexes)))]
        if index < 0:
            index += len(self.indexes)
        if index < 0 or index >= len(self.indexes):
            raise IndexError(index)
        return self.base[self.indexes[index]]

    def __iter__(self):
        for i in self.indexes:
            yield self.base[i]


# function that returns the frames of the video at the given path
def read_video(video_path, lazy=True):
    if lazy:
        return LazyVideoFrames(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
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