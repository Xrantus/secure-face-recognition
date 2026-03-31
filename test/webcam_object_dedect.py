import cv2
import time
from ultralytics import YOLO
import os

class YOLO11nDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.3):

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.cap = None

        # Performance metrics
        self.fps = 0.0
        self.prev_time = 0.0
        self.frame_count = 0

        self.load_model()


    def load_model(self):
        """Load YOLO11n model"""
        print("Loading YOLO11n Model...")
        print("  Optimized for accuracy with fewer parameters")

        try:
            if os.path.exists(self.model_path):
                print(f"  Using local model: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"  Downloading: {self.model_path}")
                self.model = YOLO(self.model_path)

            print("  Model loaded successfully!")
            print(f"  Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            print(f"  Confidence threshold: {self.conf_threshold}")

        except Exception as e:
            print(f"  Failed to load model: {e}")
            print("  Trying yolov8n.pt as fallback...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.model_path = 'yolov8n.pt'
                print("  Fallback model loaded")
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                self.model = None

    def setup_webcam(self, width=640, height=480, fps=20):
        self.cap = cv2.VideoCapture(0)

        # Raspberry Pi optimized settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def add_overlay(self, frame):
        # FPS info
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run_detection(self, show_window=True, save_video=False):
        if self.model is None:
            print("Model not loaded!")
            return

        print("\nStarting YOLO11n Webcam Detection...")
        print("=" * 40)
        print("Features:")
        print("  • Optimized YOLO11n model")
        print(f"  • Confidence threshold: {self.conf_threshold}")
        print("  • Raspberry Pi optimized settings")
        print("=" * 40)
        print("Press 'q' to quit")

        # Setup webcam for Raspberry Pi
        self.setup_webcam()

        # Video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"yolo11n_webcam_{timestamp}.mp4"
            out = cv2.VideoWriter(output_path, fourcc, 20.0,
                                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            print(f"Recording video: {output_path}")

        # Ana döngü
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # FPS hesaplama
                current_time = time.time()
                if self.prev_time > 0:
                    self.fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                # Run detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                annotated_frame = results[0].plot()

                # Add overlay
                final_frame = self.add_overlay(annotated_frame)
                self.frame_count += 1

                # Show window
                if show_window:
                    cv2.imshow('YOLO11n Detection', final_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Save video
                if save_video:
                    out.write(final_frame)

        except KeyboardInterrupt:
            print("\nStopped by user")

        except Exception as e:
            print(f"\nError: {e}")

        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if save_video and 'out' in locals():
                out.release()
                print(f"Video saved: {output_path}")
            cv2.destroyAllWindows()

            # Final stats
            print(f"\nFinal Stats:")
            print(f"  Frames processed: {self.frame_count}")
            print(f"  Average FPS: {self.fps:.1f}")
            print(f"  Model: {self.model_path}")


def main():

    # Create detector
    detector = YOLO11nDetector(model_path='yolo11n.pt', conf_threshold=0.3)

    # User options
    print("\nOptions:")
    show_window = input("Show window? (Y/n): ").lower() != 'n'
    save_video = input("Save video? (Y/n): ").lower() != 'n'

    input("\nPress Enter to start...")

    # Start detection
    detector.run_detection(show_window=show_window, save_video=save_video)

if __name__ == "__main__":
    main()
