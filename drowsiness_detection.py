import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime
import time

# Define constants
FPS = 30  # Frames per second
WARNING_DURATION = 2  # Duration to show warning (in seconds)
QUEUE_DURATION = 2  # Duration to store data in queue (in seconds)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # Yawn detection threshold -> changed to around 1 second for demo
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # Drowsy threshold -> define all sleep as 'drowsy'
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # Head movement detection threshold

def play_alarm(sound_file, duration):
    # Initialize Pygame mixer
    pygame.mixer.init()
    # Load sound file
    alarm_sound = pygame.mixer.Sound(sound_file)
    # Play sound (limited to specified time)
    alarm_sound.play(loops=0, maxtime=duration)  # Use loop for repeated play of 8-second sound snippet

def trigger_alarm(trigger, sound_file, duration):
    if trigger:
        print("Alarm is triggered!")
        play_alarm(sound_file, duration)
    else:
        print("Alarm is not triggered.")

def get_webcam_fps():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access webcam.")
        return None
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)
    return model

def webcam_detection(model, fps):
    queue_length = int(fps * QUEUE_DURATION)
    drowsy_threshold_frames = int(fps * 0.8)  # Strong drowsiness -> defined as drowsy
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access webcam.")
        return
    prev_time = time.time()

    while True:  # Process each frame
        ret, frame = cap.read()  # Read frame
        if not ret:
            print("Failed to retrieve frame.")
            break

        current_time = time.time()
        fps_real = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {int(fps_real)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Preprocess image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
        results = model.predict(source=[img], save=False)[0]  # Get model prediction results

        # Visualize results and extract object information
        detected_event_list = []  # Initialize list for detected events
        current_eye_closed = False
        current_yawn = False
        current_head_event = False

        for result in results:  # Iterate over detected objects
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                confidence = confs[i]
                label = int(classes[i])

                # Print object information
                print(f"Detected {model.names[label]} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")

                if confidence > 0.5:  # Only show objects with confidence > 0.5
                    label_text = f"{model.names[label]} {confidence:.2f}"

                    # Default color: green
                    color = (0, 255, 0)

                    # Check eye-closed status (assume labels 0, 1, 2 are eye closed)
                    if label in [0, 1, 2]:
                        current_eye_closed = True

                    # Check head up/down status (labels 4, 5 are head states)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # Set yellow
                        current_head_event = True

                    # Check yawn status (label 8)
                    if label == 8:
                        color = (0, 255, 255)  # Set yellow
                        current_yawn = True

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Append current eye-closed status to queue
        eye_closed_queue.append(current_eye_closed)

        # Append yawn event to queue
        yawn_queue.append(current_yawn)

        # Append head movement event to queue
        head_queue.append(current_head_event)

        # Determine drowsiness state based on recent eye closure
        eye_closed_count = sum(eye_closed_queue)
        if eye_closed_count >= drowsy_threshold_frames:
            detected_event_list.append('drowsy')
            drowsy_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 3000)  # 3 seconds alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)

        # Trigger warning if yawning occurs frequently
        yawn_count = sum(yawn_queue)
        if yawn_count >= yawn_threshold_frames:
            detected_event_list.append('yawn')
            yawn_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # 1 second alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            yawn_queue.clear()

        # Trigger warning if head movement occurs frequently
        head_event_count = sum(head_queue)
        if head_event_count >= head_threshold_frames:
            detected_event_list.append('head_movement')
            head_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # 1 second alarm
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            head_queue.clear()

        # Reset alarm if no events detected
        if eye_closed_count < drowsy_threshold_frames and yawn_count < yawn_threshold_frames and head_event_count < head_threshold_frames:
            alarm_end_time = None

        # Current time
        current_time = datetime.datetime.now()

        # Change bounding box color based on drowsiness state
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                label = int(classes[i])
                if label in [0, 1, 2]:  # Only change color for eye closed state
                    if 'drowsy' in detected_event_list:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, model.names[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display warning messages
        font_scale = 0.75
        font_thickness = 2

        if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)
        if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Yawning Detected!', (50, 50), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
        if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Head Up/Down Detected!', (50, 100), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)

        cv2.imshow('YOLOv8 Webcam Object Detection', frame)  # Show output
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fps = get_webcam_fps()
    print(f"Webcam FPS: {fps}")
    model_path = 'best.pt'  # Path to model
    model = load_model(model_path)
    webcam_detection(model, fps)
