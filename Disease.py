import os
import glob
import random
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from Detection import ObjectDetection
import matplotlib.pyplot as plt
from keras.preprocessing import image

class NailDiseaseDetector:
    def __init__(
        self,
        model_path,
        target_size,
        disease_image_dir ='./Disease_image',
        video_path='./resources/hand_video.MOV'
        ):
        self.class_labels = ['Beaus Line', 'Muehrckes Lines', 'Koilonychia', 'Healthy Nail', 'Onychogryphosis', 'Pitting']
        
        self.model = load_model(model_path)
        self.target_size = target_size
        self.disease_image_folder = disease_image_dir
        self.video_path = video_path

    def predict_result(self, img):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
        img_array = img_array / 255.0

        img_np = img_array[0] # (224, 224, 3)
        img_np_visual = (img_np * 255).astype(np.uint8)
        img_np_visual = cv2.cvtColor(img_np_visual, cv2.COLOR_BGR2RGB)
        plt.imshow(img_np_visual)
        plt.title('Model Input Image (img_input)')
        plt.axis('off')
        plt.show()

        preds = self.model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_label = self.class_labels[pred_idx]
        confidence = preds[0][pred_idx]
        print(f"predict result: {pred_label} (confidence: {confidence:.3f})")
        return pred_label, confidence

    def find_example_image(self, pred_label):
        pattern = os.path.join(self.disease_image_folder, f"*{pred_label}*.*")
        matched_files = glob.glob(pattern)

        if not matched_files:
            raise FileNotFoundError(f"'No {pred_label}' label image in '{self.disease_image_folder}'.")

        example_path = random.choice(matched_files)
        example_img = cv2.imread(example_path)
        if example_img is None:
            raise IOError("Failed to import example image")

        return example_img

    def detect_and_predict(self, cropped_images):
        results_list = []
        print("\n-------Image Predicting---------")
        for one in cropped_images:
            pred_label, confidence = self.predict_result(one)
            results_list.append((pred_label, confidence))
        return results_list

# -------
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Variable
    model_path='./model/ResNet50V2_model.h5'
    target_size=(224, 224)
    disease_image_dir ='./Disease_image',
    video_path='./resources/hand_video.MOV'

    nail_detector = ObjectDetection()
    disease_detector = NailDiseaseDetector(model_path=model_path, target_size=target_size)
    detect_result = nail_detector.extract_frames(disease_detector.video_path, confidence=60)
    if detect_result is None:
        raise ValueError("Failed to extract image frame in video")

    cropped_images = nail_detector.crop_objects(detect_result, target_shape=disease_detector.target_size, enlargement=False)
    results = disease_detector.detect_and_predict(cropped_images)
    print("\n--- Nail disease prediction results ---")
    for idx, (label, conf) in enumerate(results, 1):
        print(f"Nail {idx}: {label} (confidence: {conf:.3f})")

    try:
        if results:
            example_images = []
            for result in results:
                sample_label = result[0]
                example_img = disease_detector.find_example_image(sample_label)
                example_images.append(example_img)

            '''for idx, (img, label) in enumerate(zip(example_images, [r[0] for r in results])):
            cv2.imshow(f"Example {idx+1}: {label}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
        else:
            print("No prediction result.")

    except Exception as e:
        print("Error:", e)