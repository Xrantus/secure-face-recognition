"""Backend Baglanti Test Betigi

Bu dosya, kamera kullanmadan sadece RPi'nin Spring Boot backend'ine atmasi gereken 
istekleri (fetch embeddings, send log vb.) simule etmek icin kullanilir.
Kamerayi acip beklememek adina gelistirme (development) asamasinda kolaylik saglar.
"""

import time
from backend_client import fetch_and_save_embeddings, send_access_log, sync_offline_logs, _save_log_offline

def run_tests():
    print("=== BACKEND ILETISIM TESTI BASLIYOR ===\n")
    
    # Test 1: Yuz Vektoru Cekme (GET /api/embedding/all-active)
    print("--- Test 1: Baslangic Senkronizasyonu (Yuz Vektorleri Cekiliyor) ---")
    print("Aciklama: Spring Boot'tan kayitli yuzler cekilmeye calisiliyor...")
    # Sonuclari "test_known_faces.npz" dosyasina kaydeder
    result = fetch_and_save_embeddings("test_known_faces.npz")
    if result is not None:
        embs, names = result
        print(f"-> BASARILI: Toplam {len(names)} personelin yuz verisi alindi.")
        print(f"-> Gelen isimler: {names}")
    else:
        print("-> BASARISIZ: Veri cekilemedi. Backend'in calistigindan veya IP adresinin dogru oldugundan emin olun.")
        
    print("\n--------------------------------------------------------------\n")

    # Test 2: Anlik Gecis Logu Gonderme (POST /api/access-logs)
    test_personnel_id = "Ahmet-Yilmaz-123"
    print("--- Test 2: Anlik Gecis Logu (Kamera Yuz Tespit Etti) ---")
    print(f"Aciklama: Backend'e '{test_personnel_id}' isimli personel icin log atiliyor...")
    send_access_log(test_personnel_id)
    print("Not: Konsolda 'Anlik gecis logu gonderildi' yaziyorsa Spring Boot tarafina istek basariyla ulasmistir.")
    
    print("\n--------------------------------------------------------------\n")
    
    # Test 3: Cevrimdisi Log Senkronizasyonu (POST /api/access-logs/batch)
    print("--- Test 3: Cevrimdisi Log Senkronizasyonu (Offline -> Online) ---")
    print("Aciklama: Internetin kopuk oldugu senaryo simule ediliyor...")
    
    # Sistemi kandirmak icin manuel olarak iki adet offline log ekliyoruz
    _save_log_offline({"personnelId": "Mehmet-Offline", "accessTime": "2026-05-18T12:00:00"})
    _save_log_offline({"personnelId": "Ayse-Offline", "accessTime": "2026-05-18T12:05:00"})
    
    print("-> 2 adet sahte cevrimdisi (offline) log olusturuldu.")
    print("-> Simdi internetin geri geldigi varsayilip toplu yollama deneniyor...")
    time.sleep(1) # Ekranda okumayi kolaylastirmak icin kucuk bir bekleme
    sync_offline_logs()
    print("Not: Konsolda 'Cevrimdisi loglar basariyla senkronize edildi' yaziyorsa Spring Boot toplu loglari alabilmistir.")
    
    print("\n=== TEST TAMAMLANDI ===")

if __name__ == "__main__":
    run_tests()
