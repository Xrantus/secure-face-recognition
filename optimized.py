import cv2
import time
from ultralytics import YOLO

# --- SETTINGS ---
# Determines how often object detection is performed.
# Increasing the value increases FPS but reduces the freshness of detections.
FRAME_SKIP = 3

# Load our model (nano version is best for speed)
model = YOLO('yolo11n.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Increase performance by reducing resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Webcam could not be initialized.")
    exit()

# Variables
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame could not be read.")
        break

    # Flip the image horizontally (fixes mirror effect)
    frame = cv2.flip(frame, 1)

    # Increment frame counter for FPS calculation
    fps_frame_count += 1

    # Run the model only at specified intervals
    if frame_counter % FRAME_SKIP == 0:
        # Increase speed by running the model with smaller image size
        results = model(frame, stream=True, imgsz=320, verbose=False)
        last_results = list(results)  # Store results

    # If we have previous results, draw them on the screen
    if last_results:
        for r in last_results:
            boxes = r.boxes
            for box in boxes:
                # Get class ID
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Uncomment to draw only 'person' class:
                # if class_name == 'person':
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Write label
                label = f'{class_name} {box.conf[0]:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Calculate and print FPS to console
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"Current FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    cv2.imshow('Object Detection Webcam', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()