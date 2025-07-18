import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from roboflow import Roboflow

class ObjectDetection:
    def __init__(self):
        print("-------------------------------")
        print('\nInitializing Detection Model...\n')
        self.rf = Roboflow(api_key="JvYFbwrmQKx1JaMybEKR")
        self.project = self.rf.workspace().project("nail-segmentation-odcgv")
        self.model = self.project.version(2).model

    def detect_nails(self, original_image, confidence):
        result = self.model.predict(original_image, confidence=confidence).json()
        predictions = result.get('predictions', [])

        if 3 < len(predictions) < 6:
            bounding_boxes = []
            bounding_image = original_image.copy()
            for det in predictions:
                x1 = int(det['x'] - det['width'] / 2)
                y1 = int(det['y'] - det['height'] / 2)
                x2 = int(det['x'] + det['width'] / 2)
                y2 = int(det['y'] + det['height'] / 2)
                cv2.rectangle(bounding_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                bounding_boxes.append((x1, x2, y1, y2))
            # self.visualize(bounding_image, title='Detections')
            return [original_image, predictions, bounding_boxes]
        else:
            return None
        
    def extract_frames(self, video_path, confidence, half_frames=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None
        
        print("\n-------Video information-------")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution: {width} x {height}")
        print(f"Total frames: {total_frames}")
        print(f"Frame per second (FPS): {fps:.2f}")
        print(f"Duration (sec): {duration_sec:.2f}")

        print("\n-------Detecting---------------")
        frame_num = 0
        detect_results = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_this_frame = (frame_num % 2 == 0) if half_frames else True

            if process_this_frame:
                detect_result = self.detect_nails(frame, confidence=confidence)
                if detect_result:
                    original_image, predictions, bounding_boxes = detect_result
                    predictions_length = len(predictions)
                    print(f"In frame {frame_num+1}, {predictions_length} nails detected.")
                    print(f"Accuracies: {[round(predictions[i]['confidence'], 2) for i in range(predictions_length)]}")
                    detect_results[frame_num] = detect_result

            frame_num += 1

        cap.release()
        if not detect_results:
            print("No object was detected.")
            return None
        else:
            print(len(detect_results))
            return detect_results

    @staticmethod
    def visualize(image, title='Image'):
        plt.figure(figsize=(10,6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()

    @staticmethod
    def crop_objects(result, target_shape, enlargement=True):
        cropped_images = []
        target_h, target_w = target_shape
        frame, pred_result, boxes = result

        for idx, pred in enumerate(pred_result):
            x1, x2, y1, y2 = boxes[idx]
            crop = frame[y1:y2, x1:x2].copy()
            orig_h, orig_w = crop.shape[:2]

            if enlargement:
                ratio = min(target_w / orig_w, target_h / orig_h)
                new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
                crop_resize = cv2.resize(crop, (new_w, new_h))
                result = np.full((target_h, target_w, 3), (255, 255, 255), dtype=np.uint8)
                x = (target_w - new_w) // 2
                y = (target_h - new_h) // 2
                result[y:y+new_h, x:x+new_w] = crop_resize
                cropped_images.append(result)
            else:
                result = np.full((target_h, target_w, 3), (255, 255, 255), dtype=np.uint8)
                x = (target_w - orig_w) // 2
                y = (target_h - orig_h) // 2

                paste_crop = crop
                px, py = x, y
                if x < 0 or y < 0:
                    start_x = max(0, -x)
                    start_y = max(0, -y)
                    end_x = start_x + target_w
                    end_y = start_y + target_h
                    paste_crop = crop[start_y:end_y, start_x:end_x]
                    px = max(0, x)
                    py = max(0, y)

                result[py:py+paste_crop.shape[0], px:px+paste_crop.shape[1]] = paste_crop
                cropped_images.append(result)

        return cropped_images
    
# -------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Variable
    VIDEO_PATH = './resources/hand_video_3.mp4'
    target_shape = (224, 224)
    confidence = 0.5
    half_frames = False

    detector = ObjectDetection()
    results = detector.extract_frames(VIDEO_PATH, confidence=confidence, half_frames=half_frames)
    if results is not None:
        for result in results.values():
            cropped_images = detector.crop_objects(result, target_shape=target_shape, enlargement=True)

            concat_img = np.hstack(cropped_images)
            detector.visualize(concat_img, title='Connected Objects')

    end_time = time.time()
    print(f"Run time: {end_time - start_time:.2f} sec")