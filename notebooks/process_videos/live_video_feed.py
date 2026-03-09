import cv2
import mediapipe as mp
import yaml
import pathlib

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://github.com/google-ai-edge/mediapipe/

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

script_dir = pathlib.Path(__file__).parent

# ===========================================================
# ======================= SETTINGS ==========================
show_video = True
show_data = True
flip_image = False # when your webcam flips your image, you may need to re-flip it by setting this to True

cap = cv2.VideoCapture(filename=str(script_dir.joinpath("../demo_data/demo_video_rotate.mp4"))) # Video
# cap = cv2.VideoCapture(index=0) # Live from camera (change index if you have more than one camera)


# ===========================================================

# the names of each joint ("keypoint") are defined in this yaml file:
with open(script_dir.joinpath("keypoint_mapping.yml"), "r") as yaml_file:
    mappings = yaml.safe_load(yaml_file)
    KEYPOINT_NAMES = mappings["face"]
    KEYPOINT_NAMES += mappings["body"]


success = True
# find parameters for Pose here: https://google.github.io/mediapipe/solutions/pose.html#solution-apis
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and success:
        success, image = cap.read()
        if not success:
            break

        if flip_image:
            image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, but MediaPipe requires RGB
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image
        if show_video:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)

        # press ESC to stop the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # =================================
        # ===== read and process data =====
        if show_data and results.pose_landmarks is not None:
            result = f"timestamp: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.1f} seconds\n"
            
            for joint_name in ["nose", "right_wrist", "left_wrist"]: # you can choose any joint listed in `KEYPOINT_NAMES`
                joint_data = results.pose_landmarks.landmark[KEYPOINT_NAMES.index(joint_name)]
                result += f"   {joint_name:<12s} > (x: {joint_data.x:.2f}, y: {joint_data.y:.2f}, z: {joint_data.z:.2f}) [{joint_data.visibility*100:3.0f}% visible]\n"
            print(result)
        # ==================================
    cap.release()
