import pathlib

import cv2
import mediapipe as mp
from helpers import data_to_csv as dtc
import time

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://github.com/google-ai-edge/mediapipe/

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

# ===========================================================
# ======================= SETTINGS ==========================
show_video = True
flip_image = False #  when your webcam flips your image, you may need to re-flip it by setting this to True
cap = cv2.VideoCapture("../demo_data/demo_video_rotate.mp4")  # Video

result_csv_filename = f"../demo_data/csv_results/csv_file_{current_time}.csv"
# ===========================================================

csv_writer = dtc.CSVDataWriter()
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

        # Draw the pose annotation on the image.
        if show_video:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        # process data
        csv_writer.read_data(data=results.pose_landmarks, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC))

csv_writer.to_csv(result_csv_filename)
cap.release()
