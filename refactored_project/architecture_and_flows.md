# Yüz Tanıma Sistemi - Mimari ve Akış Dokümanı

Bu doküman, Raspberry Pi (Edge/Uç cihaz) ile Spring Boot (Merkezi Sunucu) arasındaki entegrasyonu, sistem mimarisini ve aralarındaki veri akışlarını açıklamaktadır.

## 1. Sistem Bileşenleri

### 1.1. Raspberry Pi (Edge Device)
Yüz algılama (YOLO) ve vektör çıkarma (Face Recognizer) gibi ağır işlem yüklerini kendi üzerinde (lokalde) gerçekleştirir. Ana mantık `run_system.py` tarafından yönetilir.
*   **Kamera Modülü:** Görüntü akışını yakalar (Picamera2 veya OpenCV).
*   **YOLO & FaceRecognizer:** Görüntüden yüzü tespit eder ve 512 boyutlu sayısal bir vektör (embedding) çıkarır.
*   **Lokal Veritabanı (Cache):** RAM'de tutulan ve diske `.npz` olarak yedeklenen hızlı eşleştirme havuzu.
*   **Backend Client:** Spring Boot sunucusuna HTTP üzerinden veri gönderen (ve çeken) istemci (`backend_client.py`).
*   **FastAPI Sunucusu:** RPi üzerinde 8000 portunda çalışan ve Backend'den gelen talepleri (`/reload`, `/generate`) karşılayan minik API sunucusu.

### 1.2. Spring Boot (Merkezi Sunucu - Backend)
Tüm sistemin beyni ve kayıt merkezidir. İlişkisel bir veritabanı (SQL) kullanır.
*   **Kullanıcı ve Rol Yönetimi:** Personel bilgileri, izinleri ve yetkileri burada tutulur.
*   **Log Yönetimi:** RPi'den gelen geçiş loglarını kabul eder, kaydeder ve raporlanabilir hale getirir.
*   **Senkronizasyon:** RPi cihazını güncel tutmak için bir API (`/api/embedding/all-active`) sunar ve gerektiğinde RPi'ye komut (webhook) gönderir.

---

## 2. Temel Sistem Akışları

Sistem, internetin kopması veya anlık olarak sunucunun yavaşlaması durumlarında cihazın çökmeyip çalışmaya devam etmesi için **"Eventual Consistency" (Gecikmeli Tutarlılık)** ve **"Offline-First" (Önce Çevrimdışı)** mantığına uygun tasarlanmıştır.

### Akış A: Sistemin Başlatılması (Boot Akışı)
RPi cihazı fişe takıldığında veya `run_system.py` başlatıldığında gerçekleşir:
1.  RPi ayağa kalkar, YOLO ve FaceRecognizer modellerini belleğe yükler.
2.  Arka planda (Thread içinde) `backend_client.py` üzerinden Backend'e bağlanır (`/api/embedding/all-active`).
3.  Backend, yetkili personelin `ID/İsim` ve `Vektör` bilgilerini JSON olarak RPi'ye gönderir.
4.  RPi bu verileri alıp, ağ koptuğunda kullanabilmek için diske `.npz` dosyası (Önbellek) olarak kaydeder.
5.  Aynı veriyi RAM'e (`db_state`) yükler. Sistem kullanıma hazırdır.

### Akış B: Anlık Tanıma ve Geçiş (Online Akış)
Ağ bağlantısı sorunsuzken, kapıdan birisi geçtiğinde gerçekleşir:
1.  Kamera yüzü görür, YOLO koordinatları çizer, model yüz vektörünü çıkarır.
2.  RPi, bu vektörü RAM'indeki listeyle karşılaştırır (Kosinüs benzerliği vb.).
3.  Eşleşme bulunursa, RPi `[BASARILI]` logunu üretir.
4.  RPi anında Backend'in `/api/access-logs` endpointine `POST` isteği atarak, "Kişi X şu saatte kapıdan geçti" bilgisini iletir.
5.  Backend bunu SQL veritabanına kaydeder.

### Akış C: Çevrimdışı Tanıma (Offline Akış)
İnternet bağlantısı veya Backend sunucusu koptuğunda gerçekleşir:
1.  Tanıma işlemleri RPi'nin RAM'indeki lokal veriyle sorunsuz devam eder (Kapı açılmaya devam eder).
2.  RPi geçiş bilgisini Backend'e göndermeye çalışır ancak Timeout/Hata alır.
3.  Hata alındığında RPi geçiş verisini kaybetmez, bunu `offline_logs.json` adlı yerel dosyaya yazar.
4.  Ağ kapalı olduğu sürece tüm geçişler bu dosyada birikir.

### Akış D: Bağlantı Kurtarma ve Toplu Gönderim (Sync Akışı)
Ağ bağlantısı geri geldiğinde sistemin kendini toparlama akışıdır:
1.  Ağ geri geldikten sonra *ilk başarılı tanıma/geçiş* anında, RPi önce o anki geçişi gönderir.
2.  Anlık geçiş başarılı bir HTTP 200 kodu ile Backend'e ulaştığında, RPi ağın geldiğini anlar.
3.  RPi hemen `sync_offline_logs()` fonksiyonunu tetikler ve `offline_logs.json` dosyasını okur.
4.  Tüm biriken logları Backend'in `/api/access-logs/batch` endpointine toplu halde (array olarak) iletir.
5.  Backend bu geçmiş logları veritabanına yazar. RPi onay alınca lokaldeki `offline_logs.json` dosyasını temizler.

### Akış E: Yeni Personel Ekleme (Webhook / Reload Akışı)
Backend paneli üzerinden sisteme yepyeni biri eklendiğinde RPi'nin yeniden başlatılmasına gerek kalmadan çalışmasını sağlayan akıştır:
1.  Sistem yöneticisi, Backend üzerinden yeni bir personel ekler veya siliş işlemi yapar.
2.  Backend, RPi'nin IP'sine (8000 portundaki FastAPI'ye) bir `/reload` HTTP POST isteği fırlatır.
3.  RPi'nin FastAPI sunucusu bu isteği alır ve kamera döngüsünü dondurmadan (Thread-safe şekilde) arka planda **Akış A'daki (Boot Akışı)** veri çekme işlemini tetikler.
4.  Yeni vektörler Backend'den iner, diske ve RAM'e yazılır.
5.  Yeni kullanıcı anında kameraya yüzünü gösterdiğinde tanınır.

### Akış F: Fotoğraftan Vektör Çıkarma (Generate Akışı)
Backend tarafında yöneticinin yüklediği bir resmin sisteme kaydedilmeden önce RPi'nin AI modelini kullanarak sayısal vektöre (embedding) dönüştürülmesi akışıdır:
1.  Yönetici Backend paneline bir fotoğraf yükler.
2.  Backend, bu fotoğraf dosyasını `multipart/form-data` olarak RPi'nin `/generate` endpointine gönderir.
3.  RPi bu resmi alır, YOLO ile içindeki yüzü bulur ve 512 boyutlu vektörü hesaplar.
4.  Hesaplanan vektör, JSON formatında Backend'e geri döndürülür.
5.  Backend, personeli ve bu sayısal vektörü kendi ilişkisel veritabanına kaydeder. Sonrasında Akış E (Reload) tetiklenir.

---

## 3. Güvenlik ve Thread (İş Parçacığı) Yönetimi
*   **Inference Lock (`_inference_lock`):** Kamera döngüsü anlık olarak yüz tararken, Backend'den aniden bir fotoğraf gelip (`/generate`) modele sokulursa AI modelinin (FaceRecognizer) çakışmasını/çökmesini engellemek için kullanılır. Sadece tek bir işlem aynı anda modele erişebilir.
*   **DB Lock (`_db_lock`):** Kamera döngüsü RAM'deki veritabanından isimleri okurken, Backend'den gelen bir `/reload` komutunun RAM'i (dizileri) o esnada aniden değiştirmesi ve uygulamanın "Index Out of Bounds" tarzı hatalar verip çökmesini engeller.
*   **Log Lock (`_log_lock`):** Aynı anda iki farklı kişi kameradan geçtiğinde, offline logların `.json` dosyasına yazılması sırasında dosyanın bozulmasını engeller.
