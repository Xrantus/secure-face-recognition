import argparse
import time
from pathlib import Path
import config

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Ham Video Kaydedici (Benchmark icin)")
    parser.add_argument("--output", default="test-videos/rpi_record.h264", help="Kaydedilecek video dosyasinin yolu (.h264 uzantili olmali)")
    parser.add_argument("--fps", type=float, default=30.0, help="Kaydedilecek videonun FPS degeri")
    parser.add_argument("--duration", type=int, default=0, help="Kayit suresi (saniye). 0 ise sinirsiz (CTRL+C ile durur)")
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
        raise SystemExit("HATA: Picamera2 modulu bulunamadi! Bu script yalnizca Raspberry Pi uzerinde calisir.")

    print(f"\n[BILGI] Donanimsal (H.264) video kaydi baslatiliyor: {output_path}")
    print(f"[BILGI] Ayarlar: {config.CAMERA_CONFIG.rpi_preview_size} @ {args.fps} FPS")
    
    picam = Picamera2()
    
    # Video yapilandirmasi olustur
    # Zoom sorununu cozmek icin: Kameraya once tam genis acidan bakmasini (1640x1232 binned mode), 
    # ardindan bunu istedigimiz kucuk cozunurluge (640x480) daraltmasini soyluyoruz.
    cfg = picam.create_video_configuration(
        main={"size": config.CAMERA_CONFIG.rpi_preview_size},
        raw={"size": (1640, 1232)} # Cogu Pi kamerasi icin genis acili (Full FoV) okuma modu
    )
    picam.configure(cfg)
    
    # FPS ayarlamasi (set_controls uzerinden guvenli sekilde)
    picam.set_controls({"FrameRate": args.fps})
    picam.start()

    # H.264 Encoder (Donanimsal hizlandirmali) ve Cikti ayari
    # 5000000 = 5 Mbps (1080p ve 720p icin temiz ve kaliteli bir bitrate)
    encoder = H264Encoder(bitrate=5000000)
    output = FileOutput(str(output_path))

    if args.duration > 0:
        print(f"[BILGI] {args.duration} saniye boyunca kayit yapilacak...")
    else:
        print("[BILGI] Kayit basladi! Durdurmak icin terminalde CTRL+C yapin.\n")

    start_time = time.time()
    
    try:
        picam.start_recording(encoder, output)
        
        while True:
            # Belirli bir sure verilmis mi kontrol et
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[BILGI] {args.duration} saniyelik sure doldu.")
                break
            # CPU'yu yormamak icin ufak bekleme
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[BILGI] CTRL+C algilandi, kayit durduruluyor...")
    finally:
        # Guvenli sekilde kaydi kapat
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
    print("                 🎥 KAYIT TAMAMLANDI")
    print("-" * 50)
    print(f"  Dosya:          {output_path}")
    print(f"  Toplam Sure:    {actual_duration:.1f} Saniye")
    print(f"  Tahmini FPS:    {args.fps} FPS (Donanimsal Sabit)")
    print("=" * 50)
    print("Simdi bu videoyu benchmark scripti ile test edebilirsiniz:")
    print(f"python -m refactored_project.benchmark_video --video test-videos/{output_path.name}")

if __name__ == "__main__":
    main()
