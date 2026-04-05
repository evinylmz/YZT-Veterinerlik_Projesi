# 🐾 VetAI Pro: Klinik Karar Destek Sistemi

VetAI Pro, veteriner hekimler için tasarlanmış, hemogram (Tam Kan Sayımı - CBC) verilerini analiz eden yapay zeka destekli bir klinik karar destek sistemidir. Makine öğrenimi algoritmaları ve Google Gemini dil modelini kullanarak kan değerlerinden ön tanı üretir ve interaktif bir konsültasyon ortamı sağlar.

## 🌟 Özellikler

* **Otomatik Ön Tanı:** WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, NEUp, LYMn ve RDWCV değerlerini kullanarak karar ağacı (Decision Tree) algoritması ile Anemi ve Polistemi alt türlerini sınıflandırır.
* **Klinik Konsültasyon:** Hastanın tahlil sonuçlarına özel olarak Google Gemini AI ile entegre, dinamik bir sohbet arayüzü sunar.
* **Hastalık Bilgi Kütüphanesi:** Kritik kan değerleri ve olası hastalıklar (Demir Eksikliği, B12 Eksikliği, Polistemia Vera vb.) hakkında hızlı referans rehberi içerir.
* **Dinamik Soru Önerileri:** Hekimlere, çıkan sonuca göre sorulabilecek en mantıklı soruları otomatik olarak önerir.

## 📁 Veri Seti

Projede kullanılan `dataset_filled.csv` veri seti, çeşitli evcil hayvanların kan tahlili parametrelerini içermektedir. Karar ağacı modeli, bu veri seti üzerindeki değerlerin klinik eşiklere (referans aralıklarına) göre etiketlenmesiyle eğitilmiştir.

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

* Python 3.8 veya üzeri
* Geçerli bir Google Gemini API Anahtarı
   
