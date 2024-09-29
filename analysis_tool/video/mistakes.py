import os
from math import sqrt

import cv2
import mediapipe as mp
import numpy as np
from fer import FER

from analysis_tool.mistakes.mistakes import MistakeType, MistakeCategory
from analysis_tool.mistakes.models import Mistake
from analysis_tool.params import PROJECT_ROOT
from analysis_tool.video.video_parser import VideoParser


def get_video_mistakes(video: VideoParser) -> list[Mistake]:
    other_people_mistakes = recognize_other_people(video)
    turning_away_mistakes = detect_turning_away_and_gestures(video)
    expressions_mistakes = analyze_expressions(video)
    return [*other_people_mistakes, *turning_away_mistakes, *expressions_mistakes]


def analyze_expressions(video: VideoParser) -> list[Mistake]:
    detector = FER()
    cap = cv2.VideoCapture(video.file_path)
    mistakes = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process people every 100ms
        if frame_count % (video.fps * 0.1) != 0:
            frame_count += 1
            continue

        frame_count += 1
        emotions = detector.detect_emotions(frame)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Timestamp in seconds

        if not emotions:
            continue

        strongest_emotion = None
        max_confidence = 0
        for emotion_name, confidence in emotions[0]["emotions"].items():
            if (
                confidence > max_confidence
                and confidence > 0.9
                and emotion_name != "neutral"
            ):
                strongest_emotion = emotion_name
                max_confidence = confidence

        if strongest_emotion:
            mistakes.append(
                Mistake(
                    type=MistakeType.FACIAL_EXPRESSIONS,
                    category=MistakeCategory.VIDEO,
                    confidence=max_confidence,
                    start_ts=current_time,
                    end_ts=current_time + 1,
                    detail=strongest_emotion,
                )
            )

    cap.release()
    return mistakes


def recognize_other_people(video: VideoParser) -> list[Mistake]:
    yolo_config = os.path.join(PROJECT_ROOT, "analysis_tool", "video", "yolo_config")
    net = cv2.dnn.readNet(
        os.path.join(yolo_config, "yolov4-tiny.weights"),
        os.path.join(yolo_config, "yolov4-tiny.cfg"),
    )

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(video.file_path)
    frame_count = 0

    mistakes = []
    previous_boxes = []
    other_person_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process people every 100ms
        if frame_count % (video.fps * 0.1) != 0:
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
            average_confidence = np.mean([confidences[i] for i in indices.flatten()])
            mistakes.append(
                Mistake(
                    type=MistakeType.SECOND_PLAN_PERSON,
                    category=MistakeCategory.VIDEO,
                    start_ts=current_time,
                    confidence=average_confidence,
                )
            )

        # When second person leaves the frame
        if people_count == 1 and other_person_detected:
            other_person_detected = False
            average_confidence = np.mean([confidences[i] for i in indices.flatten()])
            mistakes[-1].confidence = (mistakes[-1].confidence + average_confidence) / 2
            mistakes[-1].end_ts = current_time

    cap.release()
    return mistakes


def detect_turning_away_and_gestures(video: VideoParser) -> list[Mistake]:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Setup for hand gesture detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(video.file_path)
    mistakes: list[Mistake] = []
    turning_away = False
    previous_hand_positions = None
    detection_started = False
    frame_count = 0
    primary_face_rect = None  # Track the primary face rectangle

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 100ms
        if frame_count % (video.fps * 0.1) != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Detect faces (Turning Away Detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Timestamp in seconds

        # Turning Away Logic
        if len(faces) > 0:
            # If this is the first detected face, save its rectangle
            if primary_face_rect is None:
                primary_face_rect = faces[0]  # Keep the first detected face as primary

            # Check if the primary face is still present
            found_primary_face = False
            for x, y, w, h in faces:
                # Compare the coordinates of the primary face with detected faces
                if np.array_equal((x, y, w, h), primary_face_rect):
                    found_primary_face = True
                    detection_started = True
                    if turning_away:
                        turning_away = False
                        mistakes[-1].end_ts = (
                            current_time  # Set end timestamp for turning away
                        )
                    break

            # If the primary face is not found anymore
            if not found_primary_face:
                primary_face_rect = None

        else:
            if not turning_away and detection_started:
                turning_away = True
                if not mistakes or current_time > mistakes[-1].end_ts:
                    mistakes.append(
                        Mistake(
                            type=MistakeType.MOVING,
                            category=MistakeCategory.VIDEO,
                            confidence=1,
                            start_ts=current_time,
                            end_ts=current_time + 1,
                        )
                    )

        # Hand Gesture Detection using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        current_hand_positions = []

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # Store the (x, y) coordinates of each hand landmark
                    current_hand_positions.append((landmark.x, landmark.y))

            # Check if hands have moved
            if previous_hand_positions is not None:
                for prev_pos, curr_pos in zip(
                    previous_hand_positions, current_hand_positions
                ):
                    movement_length = sqrt(
                        (curr_pos[0] - prev_pos[0]) ** 2
                        + (curr_pos[1] - prev_pos[1]) ** 2
                    )
                    if movement_length > 0.05 and detection_started:
                        if not mistakes or current_time > mistakes[-1].end_ts:
                            mistakes.append(
                                Mistake(
                                    type=MistakeType.MOVING,
                                    category=MistakeCategory.VIDEO,
                                    confidence=min(movement_length / 0.1, 1),
                                    start_ts=current_time,
                                    end_ts=current_time + 1,
                                )
                            )
                        break

            previous_hand_positions = current_hand_positions
        else:
            previous_hand_positions = None  # Reset if no hands detected

    cap.release()
    return mistakes
