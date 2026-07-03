import argparse
import time
from pathlib import Path
import config

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Raw Video Recorder (for Benchmarks)")
    parser.add_argument("--output", default="test_videos/rpi_record.h264", help="Destination path for recorded video file (must end in .h264)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS value for recording")
    parser.add_argument("--duration", type=int, default=0, help="Recording duration in seconds. 0 for infinite (stopped via CTRL+C)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FileOutput
    except ImportError:
        raise SystemExit("ERROR: Picamera2 module not found! This script only runs on Raspberry Pi.")

    print(f"\n[INFO] Starting hardware-accelerated (H.264) video recording: {output_path}")
    print(f"[INFO] Configuration: {config.CAMERA_CONFIG.rpi_preview_size} @ {args.fps} FPS")
    
    picam = Picamera2()
    
    # Create video configuration
    # To fix zoom issues: first instruct camera to use full field of view (1640x1232 binned mode), 
    # then scale it down to target preview resolution (640x480).
    cfg = picam.create_video_configuration(
        main={"size": config.CAMERA_CONFIG.rpi_preview_size},
        raw={"size": (1640, 1232)} # Cogu Pi kamerasi icin genis acili (Full FoV) okuma modu
    )
    picam.configure(cfg)
    
    # FPS ayarlamasi (set_controls uzerinden guvenli sekilde)
    picam.set_controls({"FrameRate": args.fps})
    picam.start()

    # H.264 Hardware-accelerated Encoder and output config
    # 5000000 = 5 Mbps (1080p ve 720p icin temiz ve kaliteli bir bitrate)
    encoder = H264Encoder(bitrate=5000000)
    output = FileOutput(str(output_path))

    if args.duration > 0:
        print(f"[INFO] Recording for {args.duration} seconds...")
    else:
        print("[INFO] Recording started! Press CTRL+C to stop.\n")

    start_time = time.time()
    
    try:
        picam.start_recording(encoder, output)
        
        while True:
            # Belirli bir sure verilmis mi kontrol et
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[INFO] Duration of {args.duration} seconds reached.")
                break
            # CPU'yu yormamak icin ufak bekleme
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C detected, stopping recording...")
    finally:
        # Safely stop recording
        try:
            picam.stop_recording()
        except Exception:
            pass
        try:
            picam.stop()
        except Exception:
            pass

    actual_duration = time.time() - start_time
    print("-" * 50)
    print("                 🎥 RECORDING COMPLETED")
    print("-" * 50)
    print(f"  File:           {output_path}")
    print(f"  Total Duration: {actual_duration:.1f} Seconds")
    print(f"  Estimated FPS:  {args.fps} FPS (Hardware Constant)")
    print("=" * 50)
    print("You can now evaluate this video using the benchmark script:")
    print(f"python -m src.benchmarks.compare_resolutions --video test_videos/{output_path.name}")

if __name__ == "__main__":
    main()
