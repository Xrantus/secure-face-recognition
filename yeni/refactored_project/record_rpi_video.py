import argparse
import cv2
import time
import threading
from pathlib import Path
from . import config

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Ham Video Kaydedici (Benchmark icin)")
    parser.add_argument("--output", default="test-videos/rpi_record.avi", help="Kaydedilecek video dosyasinin yolu")
    parser.add_argument("--fps", type=float, default=15.0, help="Kaydedilecek videonun FPS degeri")
    parser.add_argument("--duration", type=int, default=0, help="Kayit suresi (saniye). 0 ise sinirsiz (CTRL+C ile durur)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from picamera2 import Picamera2
    except ImportError:
        raise SystemExit("HATA: Picamera2 modulu bulunamadi! Bu script yalnizca Raspberry Pi uzerinde calisir.")

    print(f"\n[BILGI] Video kaydi baslatiliyor: {output_path}")
    print(f"[BILGI] Ayarlar: {config.CAMERA_CONFIG.rpi_preview_size} @ {args.fps} FPS")
    
    picam = Picamera2()
    cfg = picam.create_preview_configuration({"size": config.CAMERA_CONFIG.rpi_preview_size})
    picam.configure(cfg)
    picam.start()

    frame_width, frame_height = config.CAMERA_CONFIG.rpi_preview_size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, args.fps, (frame_width, frame_height))

    latest_frame = None
    frame_lock = threading.Lock()
    running = True

    def reader():
        nonlocal latest_frame, running
        while running:
            try:
                rgb = picam.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with frame_lock:
                    latest_frame = bgr
            except Exception:
                running = False
                break

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    while latest_frame is None and running:
        time.sleep(0.05)

    if args.duration > 0:
        print(f"[BILGI] {args.duration} saniye boyunca kayit yapilacak...")
    else:
        print("[BILGI] Kayit basladi! Durdurmak icin terminalde CTRL+C yapin.\n")

    start_time = time.time()
    frames_written = 0

    try:
        while running:
            # Belirli bir sure verilmis mi kontrol et
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[BILGI] {args.duration} saniyelik sure doldu.")
                break

            with frame_lock:
                if latest_frame is None:
                    continue
                frame_to_write = latest_frame.copy()

            out.write(frame_to_write)
            frames_written += 1

            # FPS'ye uymak icin ufak bekleme
            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n[BILGI] CTRL+C algilandi, kayit durduruluyor...")
    finally:
        running = False
        t.join(timeout=1)
        out.release()
        try:
            picam.stop()
        except Exception:
            pass

    actual_duration = time.time() - start_time
    print("-" * 50)
    print("                 🎥 KAYIT TAMAMLANDI")
    print("-" * 50)
    print(f"  Dosya:          {output_path}")
    print(f"  Toplam Sure:    {actual_duration:.1f} Saniye")
    print(f"  Toplam Kare:    {frames_written} Kare")
    print(f"  Ortalama FPS:   {frames_written / actual_duration:.1f} FPS")
    print("=" * 50)
    print("Simdi bu videoyu benchmark scripti ile test edebilirsiniz:")
    print(f"python -m refactored_project.benchmark_video --video test-videos/{output_path.name}")

if __name__ == "__main__":
    main()
