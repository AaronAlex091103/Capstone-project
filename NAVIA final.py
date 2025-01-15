import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO
import pyttsx3

# Initialize YOLO and TTS engine
yolo_model = YOLO('yolov8n.pt')
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load images and encode faces
def load_and_encode_images(image_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: No face detected in {filename}. Skipping this image.")
    
    return known_face_encodings, known_face_names

# Save a new face image
def save_new_face(frame, face_location, output_folder, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    face_image = cv2.resize(face_image, (200, 200))
    file_path = os.path.join(output_folder, f"{name}.jpg")
    cv2.imwrite(file_path, face_image)
    print(f"Saved new face: {file_path}")

# Obstacle detection and navigation logic with object labels
def detect_and_navigate_obstacles(frame):
    results = yolo_model(frame)
    frame_height, frame_width, _ = frame.shape

    directions = {"left": 0, "center": 0, "right": 0}
    obstacle_names = {"left": [], "center": [], "right": []}

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]  # Get object label
                box_center = (x1 + x2) // 2

                if box_center < frame_width // 3:
                    directions["left"] += 1
                    obstacle_names["left"].append(label)
                elif box_center > 2 * frame_width // 3:
                    directions["right"] += 1
                    obstacle_names["right"].append(label)
                else:
                    directions["center"] += 1
                    obstacle_names["center"].append(label)

    if directions["center"] > 0:
        obstacles = ", ".join(set(obstacle_names["center"]))
        tts_engine.say(f"Obstacle ahead: {obstacles}. Please stop.")
    elif directions["left"] > 0 and directions["right"] == 0:
        obstacles = ", ".join(set(obstacle_names["left"]))
        tts_engine.say(f"Obstacle on the left: {obstacles}. Move right.")
    elif directions["right"] > 0 and directions["left"] == 0:
        obstacles = ", ".join(set(obstacle_names["right"]))
        tts_engine.say(f"Obstacle on the right: {obstacles}. Move left.")
    elif directions["left"] > 0 and directions["right"] > 0:
        obstacles_left = ", ".join(set(obstacle_names["left"]))
        obstacles_right = ", ".join(set(obstacle_names["right"]))
        tts_engine.say(f"Obstacles on both sides. Left: {obstacles_left}. Right: {obstacles_right}. Proceed with caution.")
    tts_engine.runAndWait()

# Initialize known faces
IMAGE_FOLDER = "faces"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

known_face_encodings, known_face_names = load_and_encode_images(IMAGE_FOLDER)

# Start video capture
video_capture = cv2.VideoCapture(0)

# Flag to control obstacle detection
pause_obstacle_detection = False

print("Starting live face recognition and navigation. Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if any(matches):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name == "Unknown":
            tts_engine.say("Unknown person detected.")
            tts_engine.runAndWait()

            print("New face detected! Press 's' to save the face or any other key to skip.")
            if cv2.waitKey(0) & 0xFF == ord('s'):
                pause_obstacle_detection = True  # Pause obstacle detection
                print("Enter a name for the new person:")
                new_name = input().strip()
                if new_name:
                    top, right, bottom, left = [coord * 4 for coord in face_location]
                    save_new_face(frame, (top, right, bottom, left), IMAGE_FOLDER, new_name)

                    known_face_encodings, known_face_names = load_and_encode_images(IMAGE_FOLDER)
                pause_obstacle_detection = False  # Resume obstacle detection
            else:
                continue

        tts_engine.say(f"Hello, {name}.")
        tts_engine.runAndWait()

        top, right, bottom, left = [coord * 4 for coord in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Run obstacle detection only if not paused
    if not pause_obstacle_detection:
        detect_and_navigate_obstacles(frame)

    cv2.imshow("Live Face Recognition and Navigation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
