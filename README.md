# Fundamental Data Science: Prediksi Tingkat Obesitas dan Segmentasi Pengguna Media Sosial Menggunakan Machine Learning

## Deskripsi Proyek

Proyek ini merupakan tugas akhir mata kuliah **Fundamental Data Science** yang mengimplementasikan dua pendekatan utama dalam machine learning, yaitu **Supervised Learning** dan **Unsupervised Learning**, menggunakan dua dataset yang berbeda.

**Studi Kasus 1 (Supervised Learning)** berfokus pada pembangunan model klasifikasi untuk memprediksi tingkat obesitas seseorang berdasarkan karakteristik fisik, kebiasaan makan, aktivitas sehari-hari, dan gaya hidup. Beberapa algoritma machine learning dibandingkan untuk memperoleh model dengan performa terbaik. Model terbaik kemudian diimplementasikan ke dalam aplikasi web interaktif menggunakan **Gradio**.

**Studi Kasus 2 (Unsupervised Learning)** menerapkan algoritma **K-Means Clustering** untuk mengelompokkan pengguna media sosial berdasarkan pola penggunaan media sosial, tingkat fokus, dan produktivitas. Hasil clustering digunakan untuk mengidentifikasi karakteristik masing-masing kelompok pengguna.

---

# Struktur File

```
PROJECT/
├── Dataset/
│   ├── ObesityDataSet.csv
│   └── SocialMediaAddiction.csv
│
├── sk1-supervised learning/
│   ├── ObesityPrediction.ipynb
│   ├── app.py
│   ├── best_model_obesity.pkl
│   ├── preprocessor_obesity.pkl
│   ├── label_encoder_obesity.pkl
│   └── requirements.txt
│
├── sk2-unsupervised learning/
│   └── SocialMediaClustering.ipynb
│
└── README.md
```

| File / Folder                                         | Deskripsi                                       |
| ----------------------------------------------------- | ----------------------------------------------- |
| Dataset/ObesityDataSet.csv                            | Dataset prediksi tingkat obesitas               |
| Dataset/SocialMediaAddiction.csv                      | Dataset segmentasi pengguna media sosial        |
| sk1-supervised learning/ObesityPrediction.ipynb       | Notebook pembangunan model klasifikasi obesitas |
| sk1-supervised learning/app.py                        | Aplikasi Gradio untuk prediksi tingkat obesitas |
| sk1-supervised learning/best_model_obesity.pkl        | Model terbaik hasil pelatihan                   |
| sk1-supervised learning/preprocessor_obesity.pkl      | Pipeline preprocessing                          |
| sk1-supervised learning/label_encoder_obesity.pkl     | Label encoder untuk target                      |
| sk1-supervised learning/requirements.txt              | Daftar dependency proyek                        |
| sk2-unsupervised learning/SocialMediaClustering.ipynb | Notebook implementasi K-Means Clustering        |
| README.md                                             | Dokumentasi proyek                              |

---

# Sumber Dataset

## Studi Kasus 1 (Supervised Learning)

**Dataset:** Obesity Prediction Dataset

* Jumlah data : **2.111**
* Jumlah fitur : **17**
* Target : **NObeyesdad**

Dataset diperoleh dari Kaggle:

(https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset/data)


---

## Studi Kasus 2 (Unsupervised Learning)

**Dataset:** (https://www.kaggle.com/datasets/asifxzaman/social-media-addiction-vs-productivity-dataset)

Dataset digunakan untuk mengelompokkan pengguna media sosial berdasarkan pola penggunaan media sosial, tingkat fokus, dan produktivitas.

---

# Studi Kasus 1: Supervised Learning (Prediksi Tingkat Obesitas)

Analisis ini bertujuan membangun model klasifikasi yang mampu memprediksi tingkat obesitas seseorang berdasarkan faktor demografi, kondisi fisik, kebiasaan makan, dan gaya hidup.

## Fitur Input (X)

* Gender
* Age
* Height
* Weight
* family_history_with_overweight
* FAVC
* FCVC
* NCP
* CAEC
* SMOKE
* CH2O
* SCC
* FAF
* TUE
* CALC
* MTRANS

## Target Prediksi (y)

* NObeyesdad

Target terdiri atas beberapa kategori tingkat obesitas, mulai dari **Insufficient Weight** hingga **Obesity Type III**.

---

## Tahapan Analisis

### 1. Deskripsi Dataset

* Pemeriksaan struktur dataset
* Statistik deskriptif
* Pemeriksaan missing value
* Pemeriksaan data duplikat

### 2. Exploratory Data Analysis (EDA)

Analisis eksploratif dilakukan untuk memahami karakteristik data melalui beberapa visualisasi, antara lain:

* Distribusi kelas obesitas
* Distribusi fitur numerik
* Hubungan tinggi badan dan berat badan
* Heatmap korelasi
* Analisis fitur kategorikal terhadap target

### 3. Data Preprocessing

Tahapan preprocessing meliputi:

* Pemeriksaan missing value
* Pemeriksaan data duplikat
* Label Encoding pada variabel target
* Identifikasi fitur numerik dan kategorikal
* StandardScaler untuk fitur numerik
* OneHotEncoder untuk fitur kategorikal
* Pipeline preprocessing menggunakan ColumnTransformer

### 4. Pemilihan Model

Beberapa algoritma machine learning dibandingkan, yaitu:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Gradient Boosting Classifier
* Support Vector Machine (SVM)

### 5. Training dan Evaluasi

Model dievaluasi menggunakan beberapa metrik, yaitu:

* Accuracy
* Precision
* Recall
* F1-Score
* Cross Validation
* Confusion Matrix

### 6. Penyimpanan Model

Model terbaik beserta pipeline preprocessing disimpan menggunakan **Joblib** menjadi:

* best_model_obesity.pkl
* preprocessor_obesity.pkl
* label_encoder_obesity.pkl

---

## Hasil

Berdasarkan hasil evaluasi, **Gradient Boosting Classifier** memberikan performa terbaik dibandingkan algoritma lainnya.

Model menghasilkan **Weighted F1-Score sebesar 0.9463**, sehingga dipilih sebagai model final dan diimplementasikan ke dalam aplikasi Gradio.

---

# Studi Kasus 2: Unsupervised Learning (Segmentasi Pengguna Media Sosial)

Analisis ini bertujuan mengelompokkan pengguna media sosial berdasarkan karakteristik penggunaan media sosial dan produktivitas menggunakan algoritma **K-Means Clustering**.

Tahapan analisis meliputi:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Penanganan missing value
* Penanganan outlier menggunakan metode IQR
* Encoding data kategorikal
* Standardisasi data menggunakan StandardScaler
* Penentuan jumlah cluster optimal menggunakan Elbow Method
* Clustering menggunakan K-Means
* Evaluasi menggunakan Silhouette Score dan Davies-Bouldin Index
* Visualisasi cluster menggunakan Principal Component Analysis (PCA)

Hasil clustering menghasilkan tiga kelompok pengguna dengan karakteristik yang berbeda berdasarkan screen time, tingkat fokus, dan produktivitas.

---

# Aplikasi Prediksi (app.py)

Aplikasi dikembangkan menggunakan **Gradio** sebagai antarmuka web interaktif.

Pengguna dapat memasukkan informasi berikut:

* Gender
* Age
* Height
* Weight
* Family History with Overweight
* Frequent Consumption of High Calorie Food
* Frequency of Vegetable Consumption
* Number of Main Meals
* Consumption Between Meals
* Smoking Habit
* Daily Water Intake
* Calorie Monitoring
* Physical Activity Frequency
* Technology Usage Time
* Alcohol Consumption
* Transportation Used

Setelah seluruh data diisi dan tombol **Predict** ditekan, aplikasi akan menampilkan:

* Prediksi tingkat obesitas
* Nilai confidence dari hasil prediksi

---

# Cara Menjalankan Proyek (Supervised Learning)

## 1. Menjalankan Notebook

Notebook yang digunakan:

```
sk1-supervised learning/ObesityPrediction.ipynb
```

Langkah-langkah:

1. Buka notebook menggunakan **Google Colab**.

2. Jalankan seluruh cell secara berurutan (**Runtime → Run all**).

3. Dataset akan otomatis diunduh menggunakan **KaggleHub**, sehingga tidak diperlukan proses upload dataset secara manual.

4. Setelah proses pelatihan selesai, notebook akan menghasilkan tiga file berikut:

```
best_model_obesity.pkl
preprocessor_obesity.pkl
label_encoder_obesity.pkl
```

5. Simpan atau pindahkan ketiga file tersebut ke dalam folder:

```
sk1-supervised learning/
```

File tersebut akan digunakan oleh aplikasi Gradio saat melakukan prediksi.

---

## 2. Menjalankan Aplikasi Gradio

Masuk ke folder project:

```
cd "sk1-supervised learning"
```

Pastikan file berikut tersedia pada folder yang sama:

```
app.py
best_model_obesity.pkl
preprocessor_obesity.pkl
label_encoder_obesity.pkl
requirements.txt
```

Buat virtual environment (opsional):

```
python -m venv venv
```

Aktifkan virtual environment.

Windows:

```
venv\Scripts\activate
```

Linux/macOS:

```
source venv/bin/activate
```

Install seluruh dependency:

```
pip install -r requirements.txt
```

atau secara manual:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib gradio kagglehub
```

Jalankan aplikasi:

```
python app.py
```

Setelah berhasil dijalankan, Gradio akan menampilkan **URL lokal** yang dapat dibuka melalui browser untuk melakukan prediksi tingkat obesitas secara interaktif.

---

# Catatan Kompatibilitas

Apabila terjadi error saat memuat file model (.pkl), kemungkinan disebabkan oleh perbedaan versi **scikit-learn** antara lingkungan pelatihan dan lingkungan yang digunakan untuk menjalankan aplikasi.

Periksa versi yang digunakan dengan:

```
pip show scikit-learn
```

Apabila diperlukan, gunakan versi yang sama dengan lingkungan pelatihan:

```
pip install scikit-learn==<versi_yang_sesuai>
```

---

# Library yang Digunakan

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* joblib
* gradio
* kagglehub

---

# Hasil Akhir

## Supervised Learning

* Membangun sistem klasifikasi tingkat obesitas menggunakan beberapa algoritma machine learning.
* Melakukan perbandingan performa enam algoritma klasifikasi.
* **Gradient Boosting Classifier** dipilih sebagai model terbaik dengan **Weighted F1-Score sebesar 0.9463**.
* Model terbaik berhasil diimplementasikan ke dalam aplikasi web interaktif menggunakan **Gradio**.

## Unsupervised Learning

* Mengelompokkan pengguna media sosial menggunakan algoritma **K-Means Clustering**.
* Jumlah cluster optimal diperoleh sebanyak **3 cluster**.
* Hasil clustering berhasil mengidentifikasi kelompok pengguna dengan karakteristik produktivitas dan intensitas penggunaan media sosial yang berbeda.
│   ├── best_model_obesity.pkl
│   ├── preprocessor_obesity.pkl
│   ├── label_encoder_obesity.pkl
│   └── requirements.txt
│
├── sk2-unsupervised learning/
│   └── SocialMediaClustering.ipynb
│
└── README.md
```

| File / Folder | Deskripsi |
|---------------|-----------|
| Dataset/ObesityDataSet.csv | Dataset utama prediksi obesitas |
| Dataset/social_media_addiction.csv | Dataset utama untuk clustering pengguna media sosial |
| sk1-supervised learning/ObesityPrediction.ipynb | Notebook analisis dan pembangunan model klasifikasi obesitas |
| sk1-supervised learning/app.py | Aplikasi Gradio untuk prediksi tingkat obesitas |
| sk1-supervised learning/best_model_obesity.pkl | Model terbaik hasil pelatihan |
| sk1-supervised learning/preprocessor_obesity.pkl | Pipeline preprocessing |
| sk1-supervised learning/label_encoder_obesity.pkl | Label encoder target |
| sk1-supervised learning/requirements.txt | Daftar dependency Python |
| sk2-unsupervised learning/SocialMediaClustering.ipynb | Notebook clustering menggunakan K-Means |
| README.md | Dokumentasi proyek |

---

* 1.8075
- Hasil clustering menunjukkan pola yang berbeda antara pengguna dengan produktivitas tinggi, sedang, dan rendah berdasarkan perilaku penggunaan media sosial.
