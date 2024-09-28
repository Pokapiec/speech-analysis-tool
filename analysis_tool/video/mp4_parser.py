import difflib
import os

import cv2
import easyocr
import numpy as np

from analysis_tool.mistakes.mistakes import MistakeType
from analysis_tool.mistakes.models import Mistake
from analysis_tool.params import PROJECT_ROOT

reader = easyocr.Reader(["pl"], gpu=True)


class MP4Parser:
    SUBTITLE_REGION = (470, 830, 1450, 1080)

    def __init__(self, video_path: str):
        self._video_path: str = video_path

        cap = cv2.VideoCapture(self._video_path)

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._frame_count / self._fps  # in seconds

        cap.release()

    def extract_subtitles(self) -> str:
        cap = cv2.VideoCapture(self._video_path)
        frame_count = 0
        text_from_video = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process subtitles every 2 seconds
            if frame_count % (self._fps * 2) != 0:
                frame_count += 1
                continue

            x1, y1, x2, y2 = self.SUBTITLE_REGION
            cropped_grey_frame = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            result = reader.readtext(cropped_grey_frame)  # OCR

            extracted_text = " ".join([item[1] for item in result])

            if extracted_text:
                last_element = text_from_video[-1] if text_from_video else ""
                similarity_ratio = difflib.SequenceMatcher(
                    None, last_element, extracted_text
                ).ratio()

                # Only append text if it's significantly different
                if similarity_ratio < 0.95:
                    text_from_video.append(extracted_text)

            frame_count += 1

        cap.release()
        return " ".join(text_from_video).replace(";", "")

    def recognize_other_people(self) -> list[Mistake]:
        yolo_config = os.path.join(
            PROJECT_ROOT, "analysis_tool", "video", "yolo_config"
        )
        net = cv2.dnn.readNet(
            os.path.join(yolo_config, "yolov4-tiny.weights"),
            os.path.join(yolo_config, "yolov4-tiny.cfg"),
        )

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        cap = cv2.VideoCapture(self._video_path)
        frame_count = 0

        mistakes = []
        other_person_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process people every 100ms
            if frame_count % (self._fps * 0.1) != 0:
                frame_count += 1
                continue

            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds

            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
            net.setInput(blob)
            outs = net.forward(output_layers)

            boxes = []
            confidences = []
            people_count = 0

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Class ID 0 is for person
                    if class_id == 0 and confidence > 0.8:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])

                        # Get bounding box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        people_count += 1

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)

            # Update people_count based on NMS results
            people_count = len(indices)

            # When second person enters the frame
            if people_count > 1 and not other_person_detected:
                other_person_detected = True
                average_confidence = np.mean(
                    [confidences[i] for i in indices.flatten()]
                )
                mistakes.append(
                    Mistake(
                        type=MistakeType.SECOND_PLAN_PERSON,
                        start_ts=current_time,
                        confidence=average_confidence,
                    )
                )

            # When second person leaves the frame
            if people_count == 1 and other_person_detected:
                other_person_detected = False
                average_confidence = np.mean(
                    [confidences[i] for i in indices.flatten()]
                )
                mistakes[-1].confidence = (
                    mistakes[-1].confidence + average_confidence
                ) / 2
                mistakes[-1].end_ts = current_time

        cap.release()
        return mistakes
