""" 
Process landmark data 
A scientific class to extract the index fingertip keypoints (landmark 8) from a video using MediaPipe Hands.
"""
import cv2
import mediapipe as mp
from tqdm import tqdm
from typing import List, Tuple

class LandmarkClass:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.index_trajectory: List[Tuple[int, int]] = [] # List to store index fingertip coordinates
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode = False,
            max_num_hands = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def correct_orientation(self, frame):
        """
        Rotate the frame if needed (for iPhone MOV videos).
        Modify rotation angle depending on how your video appears.
        """
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # or ROTATE_90_COUNTERCLOCKWISE
    
    def process_video(self, video, iphone = True, draw = True):
        try:
            processed_frames = []
            while video.isOpened():
                ret, frame = video.read()
                if iphone:
                    frame = self.correct_orientation(frame)
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in tqdm(results.multi_hand_landmarks):
                        h, w, _ = frame.shape
                        INDEX_FINGER_TIP = hand_landmarks.landmark[8]
                        cx, cy = int(INDEX_FINGER_TIP.x * w), int(INDEX_FINGER_TIP.y * h)
                        self.index_trajectory.append((cx, cy))
                        if draw:
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4), self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2))

                if draw:
                    processed_frames.append(frame.copy())
            video.release()
            return processed_frames
                        
        except Exception as e:
            raise ValueError(f"[Error] Video processing failed: {e}")
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        return self.index_trajectory