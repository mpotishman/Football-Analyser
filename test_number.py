# Quick test for NumberAssigner number reading on a single crop image.
# Set CROP_PATH to any saved crop (e.g. one written into the crops/ folder), then run:
#     python test_number.py

import cv2
from numberAssigner import NumberAssigner

CROP_PATH = "/Users/michaelpotishman/Documents/University/Individual Work/GitHub/Machine Learning/Football Analyser With Yolo/crops/segment_1/track_477/segment_1_track_477_frame_734.jpg"


def main():
    # cv2.imread returns a BGR numpy array, which is exactly what predict_number expects
    crop = cv2.imread(CROP_PATH)
    if crop is None:
        raise FileNotFoundError(f"Could not read crop at {CROP_PATH}")

    number_assigner = NumberAssigner(confidence_score=0.7)

    # raw read on the WHOLE body crop (what was failing before)
    raw_full, score_full = number_assigner.read_text(crop)

    # raw read on just the upper-back number region
    number_crop = number_assigner.get_number_crop(crop)
    cv2.imwrite("number_crop_debug.jpg", number_crop)  # so you can eyeball what got fed in
    raw_region, score_region = number_assigner.read_text(number_crop)

    # final filtered result
    number, score = number_assigner.predict_number(crop)

    print(f"Crop:                {CROP_PATH}")
    print(f"Raw read (full body): {raw_full!r}  conf {score_full:.4f}")
    print(f"Raw read (number box):{raw_region!r}  conf {score_region:.4f}")
    print(f"Final number:         {number}  conf {score:.4f}")
    print("Saved the number-region crop to number_crop_debug.jpg")


if __name__ == "__main__":
    main()
