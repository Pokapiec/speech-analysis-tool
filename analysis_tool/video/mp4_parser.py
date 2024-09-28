import difflib

import cv2
import easyocr

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
                similarity_ratio = difflib.SequenceMatcher(None, last_element, extracted_text).ratio()

                # Only append text if it's significantly different
                if similarity_ratio < 0.95:
                    text_from_video.append(extracted_text)

            frame_count += 1

        cap.release()
        return " ".join(text_from_video).replace(";", "")
