import cv2
import time
from landmark import LandmarkClass
from script_video import load_video
from dynamic_calligraphy import DynamicCalligraphy

class MainWindow:
    def __init__(self, video_path: str = "webcam"):
        """
        Initializes the main window class that handles the video processing, 
        landmark detection, and dynamic calligraphy for the index finger.
        
        Args:
            video_path (str): Path to the video file or 'webcam' for live capture.
        """
        self.video_path = video_path
        self.landmark_processor = LandmarkClass()  # Landmark processor for index fingertip detection
        self.video = load_video(video_path)  # Load the video or webcam stream
        self.calligraphy = DynamicCalligraphy()  # Class to handle dynamic calligraphy creation

    def resize_with_height(self, image, height: int):
        """
        Resizes an image to a specified height while maintaining its aspect ratio.
        
        Args:
            image: The image to resize.
            height (int): The desired height for the resized image.
        
        Returns:
            The resized image with the same aspect ratio.
        """
        h, w = image.shape[:2]  # Get current height and width
        scale = height / h  # Calculate scale factor for resizing
        width = int(w * scale)  # Adjust width based on scale
        return cv2.resize(image, (width, height))  # Resize image to target height

    def interface(self, original_video_path: str, processed_frames: list, index_trajectory: list):
        """
        Combines the original video, dynamic calligraphy canvas, and processed frames.
        Displays them all in one combined window while displaying the FPS.

        Args:
            original_video_path (str): Path to the original video file.
            processed_frames (list): List of processed frames with hand landmarks.
            index_trajectory (list): List of (x, y) coordinates for the fingertip trajectory.
        """
        cap = load_video(original_video_path)  # Load the original video
        # Calculate normalization bounds for the trajectory
        x_vals = [pt[0] for pt in index_trajectory]
        y_vals = [pt[1] for pt in index_trajectory]
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        frame_idx = 0
        prev_time = time.time()  # Track the previous frame time for FPS calculation
        out = None  # Initialize the video writer for saving the output

        while cap.isOpened() and frame_idx < len(processed_frames):
            ret, original_frame = cap.read()  # Read each frame from the video
            original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)  # Correct orientation for iPhone MOV videos
            if not ret:  # Exit if there are no more frames
                break

            processed_frame = processed_frames[frame_idx]  # Get the processed frame
            current_point = index_trajectory[frame_idx] if frame_idx < len(index_trajectory) else None  # Get current fingertip position

            # Update the dynamic calligraphy canvas with the latest fingertip position
            canvas_img = self.calligraphy.update_canvas(
                current_point,
                index_trajectory[:frame_idx+1],  # Pass all points up to the current frame
                min_x, max_x, min_y, max_y  # Normalize to canvas size
            )
            
            frame_idx += 1

            # Resize all frames to the same height
            target_height = 512
            original_resized = self.resize_with_height(original_frame, target_height)
            processed_resized = self.resize_with_height(processed_frame, target_height)
            canvas_resized = cv2.resize(canvas_img, (original_resized.shape[1], target_height))

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)  # Calculate FPS based on time difference
            prev_time = current_time

            # Display FPS on the processed frame
            cv2.putText(
                processed_resized,
                f"FPS: {fps:.2f}",
                (processed_resized.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0), 2
            )

            # Concatenate all three frames (original, calligraphy, processed) horizontally
            combined = cv2.hconcat([original_resized, canvas_resized, processed_resized])

            # Initialize video writer if not already initialized
            if out is None:
                frame_width = combined.shape[1]
                frame_height = combined.shape[0]
                out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height)) 
            
            out.write(combined)  # Write the combined frame to the output video

            # Display the combined frames in a window
            cv2.imshow("Dynamic Index Finger Calligraphy", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break

        cap.release()  # Release the video capture object
        if out is not None:
            out.release()  # Release the video writer
        cv2.destroyAllWindows()  # Close all OpenCV windows

    def run(self):
        """
        Runs the video processing and display logic.
        Processes the video to extract the index fingertip trajectory and displays the result.
        """
        print("[INFO] Processing video and extracting index finger trajectory...")
        processed_frames = self.landmark_processor.process_video(self.video, draw=True)
        index_trajectory = self.landmark_processor.get_trajectory()

        if not index_trajectory:
            print("[WARNING] No index trajectory detected.")
            return

        print("[INFO] Launching visual interface...")
        self.interface(self.video_path, processed_frames, index_trajectory)