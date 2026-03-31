import cv2
import time
import statistics
from ultralytics import YOLO

# Models to be used
MODELS = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv10n': 'yolov10n.pt',
    'YOLO11n': 'yolo11n.pt',
    'YOLO12n': 'yolo12n.pt'
}

# Video file path
VIDEO_PATH = 'demo_video.mp4'

def test_video_with_models():
    """Tests multiple models on video and compares them"""

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: {VIDEO_PATH} could not be opened.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frame count: {total_frames}")
    print("=" * 50)

    # Dictionaries to store results for each model
    results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{model_name} is being tested...")

        # Load model
        model = YOLO(model_path)

        # Store data for this model
        fps_values = []
        total_objects = 0
        frame_count = 0

        # Process from the beginning of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Increment frame count
            frame_count += 1

            # Start time for FPS calculation
            start_time = time.time()

            # Perform object detection
            results_model = model(frame, stream=True, imgsz=640, verbose=False)

            # Process results
            detected_objects = 0
            for r in results_model:
                detected_objects += len(r.boxes)

            total_objects += detected_objects

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_values.append(fps)

            # Show status every 100 frames
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{total_frames} frames processed...")

        # Calculate statistics for this model
        avg_fps = statistics.mean(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        avg_objects = total_objects / frame_count

        results[model_name] = {
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'avg_objects': avg_objects,
            'total_objects': total_objects,
            'processed_frames': frame_count
        }

        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  FPS range: {min_fps:.2f} - {max_fps:.2f}")
        print(f"  Average objects per frame: {avg_objects:.2f}")
        print(f"  Total detected objects: {total_objects}")

    # Display results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<12} {'Avg.FPS':<10} {'Min FPS':<10} {'Max FPS':<10} {'Objects/Frame':<12} {'Total Objects':<12}")
    print("-" * 70)

    for model_name in results:
        data = results[model_name]
        print(f"{model_name:<12} {data['avg_fps']:<10.2f} {data['min_fps']:<10.2f} {data['max_fps']:<10.2f} {data['avg_objects']:<12.2f} {data['total_objects']:<12}")

    print("-" * 70)

    # Find best performance
    best_fps = max(results.items(), key=lambda x: x[1]['avg_fps'])
    best_object_detection = max(results.items(), key=lambda x: x[1]['avg_objects'])

    print("BEST PERFORMANCE:")
    print(f"  Highest FPS: {best_fps[0]} ({best_fps[1]['avg_fps']:.2f} FPS)")
    print(f"  Most object detection: {best_object_detection[0]} ({best_object_detection[1]['avg_objects']:.2f} objects/frame)")

    # Clean up resources
    cap.release()

if __name__ == "__main__":
    test_video_with_models()
