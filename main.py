import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import pyautogui
import time
import os

# here we're loading the pre-trained tensorflow model that detects objects
model_path = 'models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
detect_fn = tf.saved_model.load(model_path)

# this function prepares the image for input into the model
def preprocess_image(image):
    input_tensor = tf.convert_to_tensor(image)  # convert the image to a tensor
    input_tensor = input_tensor[tf.newaxis, ...]  # add a batch dimension
    return input_tensor

# this function detects faces within an image
def detect_faces(image):
    input_tensor = preprocess_image(image)  # prepare the image
    detections = detect_fn(input_tensor)  # run the detection model

    # extract bounding boxes and scores from the model's output
    bboxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    faces = []
    for i in range(len(scores)):
        if scores[i] > 0.5:  # filter out detections with low confidence
            ymin, xmin, ymax, xmax = bboxes[i]
            h, w, _ = image.shape
            # convert bounding box coordinates to pixel values
            faces.append((int(xmin * w), int(ymin * h), int((xmax - xmin) * w), int((ymax - ymin) * h)))
    return faces

# this function captures an image from the webcam and saves it to a file
def take_photo(filename):
    cap = cv2.VideoCapture(0)  # open the webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    ret, frame = cap.read()  # capture a frame
    cap.release()  # release the webcam
    if ret:
        cv2.imwrite(filename, frame)  # save the frame to a file
    else:
        print("Error: Unable to capture photo.")

# this function takes a screenshot of the current screen
def take_screenshot(filename):
    screenshot = pyautogui.screenshot()  # capture the screen
    screenshot.save(filename)  # save the screenshot to a file

# this function combines a photo and screenshot into one image
def combine_images(photo_path, screenshot_path, output_path):
    photo = Image.open(photo_path).convert("RGBA")  # open and convert the photo
    screenshot = Image.open(screenshot_path).convert("RGBA")  # open and convert the screenshot

    screenshot_width, screenshot_height = screenshot.size
    photo_width, photo_height = photo.size
    if photo_width > screenshot_width or photo_height > screenshot_height:
        photo = photo.resize((screenshot_width, screenshot_height))  # resize photo if needed

    combined = Image.new('RGBA', screenshot.size)  # create a new blank image
    combined.paste(screenshot, (0, 0))  # paste the screenshot onto the blank image
    combined.paste(photo, (0, 0), photo)  # paste the photo onto the combined image
    combined.save(output_path)  # save the combined image

# this function determines the next available image number for saving
def get_next_image_number(base_path):
    i = 1
    while os.path.exists(f"{base_path}{i}.png"):  # check if the file already exists
        i += 1
    return i

def main():
    cooldown_period = 10  # the time to wait before taking another picture
    last_capture_time = 0

    while True:
        take_photo('photo.jpg')  # capture a photo
        frame = cv2.imread('photo.jpg')  # read the photo

        faces = detect_faces(frame)  # detect faces in the photo
        if len(faces) > 1:  # check if more than one face is detected
            current_time = time.time()
            if current_time - last_capture_time >= cooldown_period:
                take_screenshot('screenshot.png')  # take a screenshot
                combined_base_path = 'combined_image'
                combined_path = f"{combined_base_path}{get_next_image_number(combined_base_path)}.png"
                combine_images('photo.jpg', 'screenshot.png', combined_path)  # combine photo and screenshot
                print(f"Image captured and combined successfully. Saved as {combined_path}")
                last_capture_time = current_time  # update the last capture time
            else:
                print(f"Cooldown not over. Waiting for {int(cooldown_period - (current_time - last_capture_time))} seconds.")
        else:
            print("Less than 2 faces detected.")

        time.sleep(1)  # wait for 1 second before repeating

if __name__ == "__main__":
    main()
