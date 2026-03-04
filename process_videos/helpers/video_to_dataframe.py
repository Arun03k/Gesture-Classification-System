import cv2
import mediapipe as mp
import process_videos.helpers.data_to_csv as dtc
import time

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://github.com/google-ai-edge/mediapipe/

def video_to_dataframe(video_filename, flip_image=False):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_filename)
    frames = dtc.CSVDataWriter(path="process_videos/keypoint_mapping.yml")
    success = True
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:
            success, image = cap.read()
            if not success:
                break

            if flip_image:
                image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)


            # process data
            frames.read_data(data=results.pose_landmarks, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()
    return frames.get_frames()
