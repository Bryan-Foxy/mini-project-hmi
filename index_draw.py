""" Process the INDEX_FINGER_TIP keypoints (landmark 8) from a video using MediaPipe Hands."""
import cv2
import numpy as np
from typing import List, Tuple

def process_index_finger_tip(ift_trajectory_list: List[Tuple[int, int]]) -> np.ndarray:
    """
    Draws the trajectory of the index finger tip on a white canvas
    to simulate in-air calligraphy.

    :param ift_trajectory_list: List of (x, y) coordinates of index tip
    :return: The resulting image (canvas with drawing)
    """
    if not ift_trajectory_list:
        raise ValueError("Empty trajectory list.")

    canvas_height, canvas_width = 512, 512
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255 
    x_vals = [pt[0] for pt in ift_trajectory_list]
    y_vals = [pt[1] for pt in ift_trajectory_list]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    def normalize(val, min_val, max_val, target_size):
        return int(((val - min_val) / (max_val - min_val + 1e-5)) * (target_size - 20)) + 10
    for i in range(1, len(ift_trajectory_list)):
        x1 = normalize(ift_trajectory_list[i - 1][0], min_x, max_x, canvas_width)
        y1 = normalize(ift_trajectory_list[i - 1][1], min_y, max_y, canvas_height)
        x2 = normalize(ift_trajectory_list[i][0], min_x, max_x, canvas_width)
        y2 = normalize(ift_trajectory_list[i][1], min_y, max_y, canvas_height)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return canvas