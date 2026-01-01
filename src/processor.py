import cv2
import os

class VideoProcessor:
    def __init__(self, detector, output_dir='output'):
        """
        Initialize the VideoProcessor.
        :param detector: An instance of a vehicle detector.
        :param output_dir: Directory where processed videos will be saved.
        """
        self.detector = detector
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_video(self, input_path):
        """
        Process a video file, detect vehicles, and save the result.
        :param input_path: Path to the input video file.
        """
        if not os.path.exists(input_path):
            print(f"Error: Input video file {input_path} not found.")
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Prepare output video writer
        output_filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, f"processed_{output_filename}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {input_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect vehicles
            results = self.detector.detect_vehicles(frame)
            
            # Draw detections on the frame
            # results[0] contains the detection for the single frame input
            annotated_frame = results[0].plot()

            # Write the frame to the output video
            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"Processing complete. Result saved to: {output_path}")
