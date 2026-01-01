import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the YOLOv8 model for vehicle detection.
        :param model_name: Name of the YOLOv8 model file.
        """
        self.model = YOLO(model_name)
        # COCO class IDs for vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect_vehicles(self, frame):
        """
        Detect vehicles in a single frame.
        :param frame: The input frame (numpy array).
        :return: Results object containing detections.
        """
        # Run inference on the frame, filtering for vehicle classes
        results = self.model(frame, classes=self.vehicle_classes, verbose=False)
        return results
