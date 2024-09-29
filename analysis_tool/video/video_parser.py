import difflib
import os
from functools import cache

import cv2
import easyocr

from analysis_tool.params import VIDEO_FILES_PATH, AUDIO_FILES_PATH


class VideoParser:
    SUBTITLE_REGION = (470, 830, 1450, 1080)

    def __init__(self, file_name: str):
        self.file_name: str = file_name
        self.file_path: str = os.path.join(VIDEO_FILES_PATH, file_name)

        cap = cv2.VideoCapture(self.file_path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps  # in seconds

        cap.release()

    @property
    @cache
    def ocr_subtitles(self) -> str:
        return self.extract_subtitles()

    def save_mp3(self) -> str:
        # -y flag is to always override files
        file_name_without_extension = "".join(self.file_name.split(".")[:-1])
        os.system(
            f"ffmpeg -y -i {self.file_path} {os.path.join(AUDIO_FILES_PATH, file_name_without_extension)}.mp3"
        )
        return f"{file_name_without_extension}.mp3"

    def extract_subtitles(self) -> str:
        reader = easyocr.Reader(["pl"])
        cap = cv2.VideoCapture(self.file_path)
        frame_count = 0
        text_from_video = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process subtitles every 2 seconds
            if frame_count % (self.fps * 2) != 0:
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
