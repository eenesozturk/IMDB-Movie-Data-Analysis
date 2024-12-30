# IMDB Film Veri Analizi Projesi

## Projeyi Geliştirenler

`[Muhammed Enes Öztürk, Ayberk Yazıkoz, Berhan Özbey, Emir Eken]`

Bu proje, IMDB filmleriyle ilgili bir veri kümesi kullanarak çeşitli makine öğrenimi algoritmalarını (KNN, MLP ve Naive Bayes) uygulamayı ve analiz sonuçlarını karşılaştırmayı amaçlar. Veri kümesi hem kategorik hem de sayısal öznitelikler içerir ve bu özniteliklerin algoritma performanslarına olan etkilerini incelemek için hazırlanmıştır.

---

## Proje Adımları

### 1. Veri Kümesinin Seçimi

Proje, IMDB filmleriyle ilgili özellikleri içeren `IMDB_Movies_Dataset.csv` adlı veri kümesi üzerinde gerçekleştirilmiştir. Veri kümesi filmlerin türleri, yönetmenleri, oyuncu kadrosu ve IMDB puanı gibi bilgiler içerir.

---

### 2. Özniteliklerin Tanıtımı

| **Öznitelik**         | **Açıklama**                           |
| --------------------- | -------------------------------------- |
| **Director**          | Filmin yönetmeni.                      |
| **Writer**            | Filmin senaristi.                      |
| **Country of Origin** | Filmin yapım yeri.                     |
| **Languages**         | Filmin dilleri.                        |
| **Runtime**           | Filmin süresi (dakika).                |
| **Release Year**      | Filmin çıkış yılı.                     |
| **Average Rating**    | Filmin IMDB üzerindeki ortalama puanı. |

---

### 3. Kategorik Özniteliklerin Dönüştürülmesi

Veri kümesindeki kategorik öznitelikler `LabelEncoder` kullanılarak sayısal değerlere dönüştürülmüştür:

- **Director**, **Writer**, **Country of Origin**, ve **Languages** öznitelikleri `LabelEncoder` ile dönüştürülmüştür.

---

### 4. Özniteliklerin Normalize Edilmesi

Tüm sayısal öznitelikler, farklı ölçekleri dengelemek için `StandardScaler` ile normalize edilmiştir. Bu adım, makine öğrenimi algoritmalarının daha iyi performans göstermesini sağlamıştır.

---

### 5. Makine Öğrenimi Algoritmaları

#### a. KNN

- KNN algoritması K=3, K=7 ve K=11 komşuluk değerleri için çalıştırılmıştır.
- Doğruluk (accuracy), kesinlik (precision), geri çağırma (recall) ve F1 skor değerleri hesaplanmıştır.

#### b. MLP

- Çok katmanlı algılayıcı (MLP) algoritması şu yapılandırmalarla çalıştırılmıştır:
  - 1 gizli katman (32 nöron),
  - 2 gizli katman (32'şer nöron),
  - 3 gizli katman (32'şer nöron).

#### c. Naive Bayes

- Naive Bayes algoritması varsayılan parametrelerle çalıştırılmıştır.

---

### 6. Sonuçların Analizi

- Algoritmaların doğruluk (accuracy), kesinlik (precision), geri çağırma (recall) ve F1 skor değerleri karşılaştırılmıştır.
- Elde edilen sonuçlar bir tablo halinde `algorithm_results.csv` dosyasına kaydedilmiştir.

---

## Nasıl Çalıştırılır?

1. Gerekli bağımlılıkları yüklemek için:
   ```bash
   pip install -r requirements.txt
   ```
2. Ana Python dosyasını çalıştırın:
   ```bash
   python src/main.py
   ```
3. Algoritma sonuçları `results/algorithm_results.csv` dosyasına kaydedilecektir.

---
