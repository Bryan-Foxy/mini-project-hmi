"""This script is designed to load video and them using OpenCV."""
import cv2

def load_video(file_path = "webcam"):
    """
    Load and display video from the specified file path.
    Args:
        file_path (str): The path to the video file.
    """
    try:
        if file_path == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(file_path)
        return cap
    except Exception as e:
        raise ValueError(f"Error loading video: {e}")