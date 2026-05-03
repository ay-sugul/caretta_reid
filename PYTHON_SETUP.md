# Python Kurulum Rehberi

## Problem
Python yüklü olmasına rağmen bash/Git Bash ortamında erişilemiyor.
- Python bulunuyor: `/c/Users/ayşegül/AppData/Local/Microsoft/WindowsApps/python`
- Fakat bash'den çalıştırılamıyor
- Muhtemelen Türkçe karakterler (ayşegül) PATH sorununa neden oluyor

## Çözüm Seçenekleri

### ✅ Seçenek 1: Python'u python.org'dan Standard Yere Yükle (Tavsiye Edilen)

1. https://www.python.org/downloads/ adresine git
2. Python 3.11 veya 3.12 indir
3. Kurulum sırasında:
   - ✅ "Add python.exe to PATH" işaretle
   - ✅ "Install for all users" seçerek C:\Python311\ gibi standart yere kur

4. Git Bash'i yeniden aç

5. Aşağıdaki komutlar çalışmalı:
```bash
python --version
python -m pip --version
```

### 📝 Seçenek 2: Cmd.exe Kullanan Batch Dosyası Oluştur

`run_tests.bat` dosyasını projede oluştur:
```batch
@echo off
cd /d C:\Users\ayşegül\Desktop\caretta_reid

:: Sanal ortam oluştur (varsa atla)
if not exist venv\ (
    python -m venv venv
)

:: Sanal ortamı aktif et
call venv\Scripts\activate.bat

:: Bağımlılıkları yükle (varsa atla)
if not exist venv\Lib\site-packages\torch (
    pip install -r requirements.txt
)

:: Testleri çalıştır
pytest tests/ -v

pause
```

Sonra Command Prompt'ta çalıştır:
```cmd
run_tests.bat
```

### 🐧 Seçenek 3: WSL2 (Windows Subsystem for Linux) Kur

WSL2 zaten yüklüyse:
```bash
wsl
cd /mnt/c/Users/ayşegül/Desktop/caretta_reid
python3 --version
```

WSL2 yoksa:
```powershell
# PowerShell'i Yönetici olarak aç ve çalıştır:
wsl --install
# Bilgisayarı yeniden başlat
# Daha sonra Ubuntu SSH yapılandırması at
```

### 💻 Seçenek 4: VS Code Python Uzantısı Kullan

1. VS Code'da **Run** → **Run Without Debugging** menüsünü aç
2. Veya **Python: Run Python File in Terminal** komutunu çalıştır
3. VS Code Python interpreter'ı otomatik bulacak

## Adımlar Sonrasında

Seçiminiz ne olursa olsun, aşağıdaki sırayla devam et:

```bash
# 1. Sanal ortam oluştur (tavsiye edilen)
python -m venv venv

# Windows'ta aktif et:
venv\Scripts\activate

# macOS/Linux'ta aktif et:
source venv/bin/activate

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. .env dosyası oluştur
cp .env.example .env

# 4. Testleri çalıştır
pytest tests/ -v

# 5. (Opsiyonel) Veritabanını doldur
python -m caretta_reid.database.embedding_store

# 6. Demo başlat
python -m caretta_reid.demo.app
```

## Sorun Giderme

### "command not found: python"
→ Seçenek 1'i dene (python.org'dan kur)

### "ModuleNotFoundError: No module named 'torch'"
→ `pip install -r requirements.txt` komutunu çalıştır

### "Permission denied"
→ Cmd.exe veya PowerShell'i **Yönetici olarak** aç

### Türkçe karakter PATH sorunları
→ WSL2 (Seçenek 3) kullan veya cmd.exe üzerinden (Seçenek 2) çalıştır

---

**Hangi seçeneği denemek istiyorsun?**
