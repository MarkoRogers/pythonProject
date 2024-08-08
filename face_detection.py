import cv2
import mediapipe as mp
import pyautogui
from PIL import Image
import time
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            xmin = int(bboxC.xmin * iw)
            ymin = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            faces.append((xmin, ymin, w, h))
            # Draw rectangle around the face
            cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 2)
            # Draw landmarks
            mp_drawing.draw_detection(image, detection)
    return faces, image


def take_photo(filename):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(filename, frame)
    else:
        print("Error: Unable to capture photo.")


def take_screenshot(filename):
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)


def combine_images(photo_path, screenshot_path, output_path):
    photo = Image.open(photo_path).convert("RGBA")
    screenshot = Image.open(screenshot_path).convert("RGBA")

    screenshot_width, screenshot_height = screenshot.size
    photo_width, photo_height = photo.size
    if photo_width > screenshot_width or photo_height > screenshot_height:
        photo = photo.resize((screenshot_width, screenshot_height))

    combined = Image.new('RGBA', screenshot.size)
    combined.paste(screenshot, (0, 0))
    combined.paste(photo, (0, 0), photo)
    combined.save(output_path)


def get_next_image_number(base_path):
    i = 1
    while os.path.exists(f"{base_path}{i}.png"):
        i += 1
    return i


def main():
    cooldown_period = 10  # Cooldown period in seconds
    last_capture_time = 0
    capturing = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        if capturing:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture video frame.")
                time.sleep(1)  # Wait before retrying
                continue

            faces, annotated_frame = detect_faces(frame)

            if len(faces) >= 2:
                current_time = time.time()
                if current_time - last_capture_time >= cooldown_period:
                    # Stop capturing and process the images
                    capturing = False
                    cap.release()  # Release the camera before processing

                    take_photo('photo.jpg')
                    take_screenshot('screenshot.png')

                    combined_base_path = 'combined_image'
                    combined_path = f"{combined_base_path}{get_next_image_number(combined_base_path)}.png"

                    combine_images('photo.jpg', 'screenshot.png', combined_path)
                    print(f"Image captured and combined successfully. Saved as {combined_path}")

                    last_capture_time = current_time  # Update last capture time
                    # Reinitialize the camera for further captures
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Error: Unable to open webcam.")
                        return
                    capturing = True  # Resume capturing
                else:
                    print(
                        f"Cooldown not over. Waiting for {int(cooldown_period - (current_time - last_capture_time))} seconds.")
            else:
                # Continue capturing frames if less than 2 faces are detected
                cv2.imshow('Face Detection', annotated_frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Make sure to release the camera properly if we are not capturing
            cap.release()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
