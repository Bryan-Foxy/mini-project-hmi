import cv2
import numpy as np
from typing import List, Tuple

class DynamicCalligraphy:
    def __init__(self, canvas_size: Tuple[int, int] = (512, 512), background_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize the dynamic calligraphy canvas with enhanced visuals and customizable background color.
        
        Args:
            canvas_size: Tuple (width, height) for the canvas size.
            background_color: Background color for the canvas, default is white (255, 255, 255).
        """
        self.canvas_size = canvas_size
        self.background_color = background_color 
        self.reset_canvas()

    def reset_canvas(self):
        """Reset the canvas to the background color."""
        self.canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)
        self.previous_point = None

    def update_canvas(self, current_point: Tuple[int, int], trajectory: List[Tuple[int, int]], 
                     min_x: int, max_x: int, min_y: int, max_y: int):
        """
        Update the canvas with the latest finger position and the full trajectory.
        
        Args:
            current_point: Current (x, y) position of the fingertip.
            trajectory: Full trajectory history.
            min_x, max_x: Range of x values for normalization.
            min_y, max_y: Range of y values for normalization.
        """
        # Normalization function to map coordinates to canvas space
        def normalize(val, min_val, max_val, target_size):
            return int(((val - min_val) / (max_val - min_val + 1e-5)) * (target_size - 20)) + 10
        
        # Draw the trajectory smoothly with a stronger ocean blue color
        if len(trajectory) > 1:
            # For every pair of consecutive points in the trajectory
            for i in range(1, len(trajectory)):
                x1 = normalize(trajectory[i - 1][0], min_x, max_x, self.canvas_size[0])
                y1 = normalize(trajectory[i - 1][1], min_y, max_y, self.canvas_size[1])
                x2 = normalize(trajectory[i][0], min_x, max_x, self.canvas_size[0])
                y2 = normalize(trajectory[i][1], min_y, max_y, self.canvas_size[1])

                cv2.line(self.canvas, (x1, y1), (x2, y2), (70, 130, 180), 3)

        # Highlight the current point (index fingertip) with a softer glowing effect
        if current_point:
            cx = normalize(current_point[0], min_x, max_x, self.canvas_size[0])
            cy = normalize(current_point[1], min_y, max_y, self.canvas_size[1])

            cv2.circle(self.canvas, (cx, cy), 10, (70, 130, 180), -1) 
            cv2.circle(self.canvas, (cx, cy), 8, (70, 130, 180), 4) 

        return self.canvas