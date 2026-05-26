# RPi ve Backend Entegrasyonu Gerçek Dünya Test Senaryoları

Bu belge, Raspberry Pi (RPi) cihazı üzerindeki yüz tanıma sistemi ile Spring Boot tabanlı backend sistemi arasındaki iletişimi gerçek dünya koşullarında test etmek için hazırlanmış senaryoları içermektedir. Bu testler "mock" (sahte) verilerle değil, her iki sistemin de (RPi tarafındaki `run_system.py` ve Backend tarafındaki Spring Boot uygulaması) aynı ağ üzerinde aktif olarak çalıştırıldığı canlı ortamlarda gerçekleştirilmelidir.

## Ön Hazırlık
1. Spring Boot uygulamasının çalışır ve erişilebilir durumda olduğundan emin olun.
2. RPi cihazının (veya lokal test ediyorsanız Mac/Win makinenin) backend sunucusuyla aynı ağda olduğundan emin olun.
3. `backend_client.py` içerisindeki `BACKEND_BASE_URL` değişkeninin gerçek backend IP ve portuna ayarlı olduğunu doğrulayın (örneğin: `http://192.168.1.100:8080`).

---

### Senaryo 1: İlk Başlatma ve Veritabanı Senkronizasyonu
**Amaç:** RPi başlatıldığında backend'den güncel personel yüz verilerinin çekilip çekilemediğini doğrulamak.
1. RPi üzerinde `run_system.py` uygulamasını başlatın (`python -m refactored_project.run_system`).
2. Backend sunucusunun aktif loglarına bakın; `/api/embedding/all-active` endpointine bir GET isteği gelmiş olmalı.
3. RPi konsolunda şu çıktıları arayın: 
   - `[Backend Client] Yuz verileri cekiliyor...`
   - `[API] Veritabani basariyla guncellendi!`
4. **Doğrulama:** RPi dizininde `db_path` config ile belirtilen `.npz` dosyasının (örneğin `face_db.npz`) oluşturulduğunu veya son değiştirilme tarihinin güncellendiğini teyit edin.

---

### Senaryo 2: Anlık Tanıma ve Başarılı Geçiş Logu Gönderimi (Online Mod)
**Amaç:** Sistem çevrimiçiyken tanınan bir kişinin geçiş logunun anında backend'e ulaşıp ulaşmadığını test etmek.
1. Her iki sistem de açık ve senkronize durumda olsun.
2. Kameraya veritabanında kayıtlı bir yüz gösterin.
3. RPi konsolunda `[BASARILI] {İsim} tespit edildi! (Skor: ...)` mesajını görün.
4. RPi konsolunda `[Backend Client] Anlik gecis logu gonderildi: {İsim}` mesajını görün.
5. **Doğrulama:** Backend veritabanında (ilgili Access Log tablosunda) anlık olarak o kişiye ve şimdiki saate ait "AUTHORIZED" durumunda yeni bir geçiş kaydı oluştuğunu teyit edin.

---

### Senaryo 3: Tanınmayan (Kayıtsız) Yüz Testi
**Amaç:** Sisteme kayıtlı olmayan bir kişinin gereksiz log gönderimi yapıp yapmadığını kontrol etmek.
1. Kameraya veritabanında bulunmayan bir kişinin yüzünü gösterin (veya telefon ekranından kayıtsız bir yüz resmi tutun).
2. RPi konsolunda `[HATA AYIKLAMA] Yuz algilandi ama eslesmedi.` mesajının çıktığını görün.
3. **Doğrulama:** Backend tarafına herhangi bir HTTP POST isteği gönderilmediğini ve veritabanına yeni bir log yazılmadığını teyit edin. (RPi konsolunda "anlık log gönderildi" yazısı çıkmamalıdır).

---

### Senaryo 4: Ağ Kesintisi ve Çevrimdışı (Offline) Log Kaydı
**Amaç:** İnternet veya backend bağlantısı koptuğunda geçişlerin lokalde saklandığından emin olmak.
1. Sistem çalışırken Spring Boot backend'i durdurun (veya RPi'nin Wi-Fi bağlantısını geçici olarak kesin).
2. Kameraya kayıtlı bir yüz gösterin.
3. RPi konsolunda yüzün başarıyla tanındığını görün.
4. Bağlantı koptuğu için RPi konsolunda şu hatayı görmelisiniz:
   - `[Backend Client] Baglanti yok. Log cevrimdisi kaydediliyor: {İsim}`
5. 1-2 kez daha kayıtlı yüz(ler) göstererek birkaç offline log oluşturun.
6. **Doğrulama:** RPi üzerinde `offline_logs.json` dosyasının oluştuğunu ve içeriğini kontrol ettiğinizde JSON dizisi şeklinde (array) bekleyen geçiş loglarının yer aldığını teyit edin.

---

### Senaryo 5: Bağlantı Kurtarma ve Toplu (Batch) Log Senkronizasyonu
**Amaç:** Ağ bağlantısı geri geldiğinde biriken çevrimdışı logların backend'e toplu olarak aktarıldığını doğrulamak.
1. Spring Boot backend'i tekrar başlatın (veya Wi-Fi'yi açın).
2. Sistem bekleyen logları hemen göndermez; *bir sonraki başarılı geçişte* tetiklenir.
3. Kameraya tekrar kayıtlı bir yüz gösterin.
4. RPi konsolunda şu akışı izleyin:
   - `[Backend Client] Anlik gecis logu gonderildi...`
   - `[Backend Client] {X} adet cevrimdisi log senkronize ediliyor...`
   - `[Backend Client] Cevrimdisi loglar basariyla senkronize edildi.`
5. **Doğrulama:** Backend loglarında `/api/access-logs/batch` endpointine toplu veri geldiğini görün. Backend veritabanında hem offline dönemdeki geçmiş geçişlerin hem de son anlık geçişin doğru saatlerle kaydedildiğini teyit edin.
6. **Doğrulama:** RPi dizinindeki `offline_logs.json` dosyasının silindiğini teyit edin.

---

### Senaryo 6: Backend Üzerinden RPi Veritabanı Güncelleme (Reload / Webhook Testi)
**Amaç:** Backend üzerinden bir kullanıcı eklendiğinde/silindiğinde RPi sistemini manuel olarak yeniden başlatmadan değişikliklerin alınabilmesi.
1. Sistemler normal çalışırken, Postman veya Frontend arayüzünü kullanarak Spring Boot backend üzerinden sisteme yeni bir personelin yüz bilgisini ekleyin (veya silin).
2. Backend, RPi'nin IP adresi ve portuna (örn: `http://192.168.1.50:8000/reload`) bir POST isteği atmalıdır. Bunu tetikleyin.
3. RPi konsolunda FastAPI sunucusundan kaynaklı `/reload` isteğinin düştüğünü ve ardından şu mesajı arayın:
   - `[API] Veritabani basariyla guncellendi!`
4. **Doğrulama:** Yeni eklenen kişinin yüzünü kameraya gösterin. Sistemi kapatıp açmadığınız halde anında tanınması gerekir.

---

### Senaryo 7: RPi API'sinden Resim Göndererek Vektör Alma Testi
**Amaç:** RPi üzerindeki FastAPI sunucusunun, dışarıdan gelen bir fotoğraftan embedding çıkarıp çıkarmadığını doğrulamak. (Bu endpoint backend'in yeni kullanıcı kaydederken vektör talep etmesi içindir).
1. RPi üzerinde sistem çalışır haldeyken, Postman'i açın.
2. Endpoint: `POST http://<RPI_IP>:8000/generate`
3. Body kısmında `form-data` seçin. Key olarak `file` yazıp türünü `File` seçin ve Value olarak bir yüz fotoğrafı (.jpg veya .png) yükleyin.
4. İsteği gönderin.
5. **Doğrulama:** RPi'nin 200 OK ile birlikte JSON formatında `{"embedding": [0.123, -0.456, ...]}` şeklinde 512 boyutlu bir dizi döndürdüğünü teyit edin.
6. **Hata Doğrulama:** Yüz olmayan veya manzara içeren bir fotoğraf yükleyin. 400 Bad Request statü kodlu `{"detail": "No face detected in the image."}` hatasını aldığınızı doğrulayın.

---

Bu 7 senaryo başarılı olduğu takdirde, RPi yüz tanıma sistemi ile Backend sistemi tam entegre, hataya dayanıklı (fault-tolerant) ve offline destekli (resilience) çalışıyor demektir.
