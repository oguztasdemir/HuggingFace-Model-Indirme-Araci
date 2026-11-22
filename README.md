# 🤗 Hugging Face Hub Analyzer & One-Click Downloader

[![CI Tests](https://github.com/oguztasdemir/HuggingFace-Model-Indirme-Araci/actions/workflows/ci.yml/badge.svg)](https://github.com/oguztasdemir/HuggingFace-Model-Indirme-Araci/actions/workflows/ci.yml)


Hugging Face platformundaki yapay zeka modellerini ve veri kümelerini (datasets) incelemek, yerel Ollama yapay zekası ile Türkçe özetler almak, bilgisayarınıza **tek tıkla indirmek** ve disk alanınızı canlı olarak yönetmek için geliştirilmiş modern bir masaüstü uygulamasıdır.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![UI Framework](https://img.shields.io/badge/UI-CustomTkinter-orange.svg)
![AI Backend](https://img.shields.io/badge/AI-Ollama%20Local-blueviolet.svg)

---

## ✨ Özellikler

- **🚀 Otomatik Bağımlılık Kurulumu:** Repoyu klonlayıp `python main.py` komutunu çalıştırdığınız an gerekli tüm kütüphaneler otomatik kontrol edilir ve eksikler `pip` vasıtasıyla kurulur.
- **⚡ Tek Tıkla Model İndirme:** Model URL'sini yapıştırıp analize bastıktan sonra **"Tek Tıkla Tüm Modeli İndir"** butonuna basarak tüm ağırlık dosyalarını ve snapshot'ı otomatik indirebilirsiniz.
- **📂 Seçmeli Dosya İndirme:** Depodaki dosyaları listeleyip arama çubuğu ile filtreleyebilir ve sadece ihtiyacınız olan dosyaları seçip indirebilirsiniz.
- **🦙 %100 Yerel Yapay Zeka Özeti (Ollama LLM):** Herhangi bir cloud API anahtarına gerek duymadan, bilgisayarınızda çalışan yerel Ollama (`http://localhost:11434`) vasıtasıyla modellerin amacını ve işlevini Türkçe özetler.
- **📦 Yerel Önbellek & Model Yöneticisi:** Bilgisayarınızda indirilmiş olan tüm Hugging Face modelleri taranır. Modellerin kapladığı alan görüntülenir, klasörde açılabilir veya tek tıkla silinerek diskte yer açılabilir.
- **🗂️ Sol Yan Panel & Canlı Disk Göstergesi:** Seçili indirme sürücüsünün (Örn: `E:\` veya `C:\`) toplam boyutu, boş alanı ve Hugging Face modellerinin kapladığı alan sol panelde canlı olarak gösterilir.
- **🛡️ Güvenlik Kontrolleri:**
  - **Sürücü Erişilebilirlik Kontrolü:** İndirme yapılmak istenen harici disk takılı değilse uyarı verir ve indirmeyi engeller.
  - **Yetersiz Disk Alanı Kontrolü:** İndirilecek model boyutu ile diskinizdeki boş alan kıyaslanır; yetersizse indirme öncesi uyarı gösterir.

---

## 💻 Kurulum ve Çalıştırma

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADI/Hugging-face-Downloader.git
cd Hugging-face-Downloader
```

### 2. Uygulamayı Başlatın
```bash
python main.py
```
> **Not:** `main.py` çalıştığı an `requirements.txt` dosyasını kontrol eder ve eksik kütüphaneleri (`customtkinter`, `huggingface_hub`, `requests`, `beautifulsoup4`) otomatik olarak kurar.

---

## 🛠️ Gereksinimler (`requirements.txt`)

- `customtkinter`
- `requests`
- `beautifulsoup4`
- `huggingface_hub`

---

## ⚙️ Yapılandırma (`settings.json`)

Ayarlar sekmesinden tercih ettiğiniz sabit indirme klasörünü (Örn: `E:\huggingface`) seçip kaydedebilirsiniz. Tercihleriniz kalıcı olarak `settings.json` dosyasına kaydedilir.

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.


---

## Yerel Testler

Projenin temel çalışma ve sözdizimi doğrulama testlerini çalıştırmak için:

```bash
pip install pytest
pytest tests -v
```
