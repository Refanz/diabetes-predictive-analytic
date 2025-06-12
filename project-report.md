# Laporan Proyek Machine Learning - Refanda Surya Saputra

## Domain Proyek

Diabetes merupakan penyakit kronis yang diidentifikasi dengan tingginya kadar gula darah. Salah satu sumber energi utama
bagi tubuh adalah glukosa/gula. Tetapi, pada pegidap diabetes glukosa tidak dapat dimanfaatkan oleh tubuh dengan
efektif.

Kadar gula darah dikelola oleh hormon insulin yang diproduksi oleh pankreas. Insulin membantu sel dalam tubuh untuk
menyerap glukosa. Oleh karena itu, kadar gula darah tetap stabil. Orang yang mengidap penyakit diabetes, pankreas tidak
dapat menghasilkan insulin, atau tubuh tidak mampu menggunakan insulin dengan baik. Akibatnya, sel-sel tubuh tidak mampu
menyerap dan mengatur glukosa menjadi energi.

Terdapat beberapa faktor yang dapat meingkatkan risiko terkena diabetes yaitu terdapat keluarga dengan riwayat diabetes,
terkena penyakit autoimun, gaya hidup yang buruk, dan konsumsi makanan sembarangan. Terdapat faktor lain yaitu jenis
kelamin. Menurut Rosita et al. (2022) menunjukkan bahwa jenis kelamin perempuan memiliki risiko 2,15 kali lebih besar
untuk terkena diabetes melitus tipe 2 dibandingkan dengan laki-laki. Kemudian, berdasarkan riset kesehatan
dasar
RISKESDAS (2018) penderita diabetes melitius pada perempuan (1,8%) lebih tinggi daripada laki-laki (1,2%) di Indonesia.

Risiko terkena diabetes pada perempuan lebih tinggi daripada laki-laki. Oleh karena itu, proyek ini fokus pada
pengembangan
model machine learning untuk memprediksi risiko perempuan terkena diabetes atau tidak. Terdapat 768 data
dengan pembagian sebanyak 500 tidak mengidap diabetes dan 268 mengalami diabetes. Kemudian untuk variabel prediksi
adalah jumlah
kehamilan, glukosa, tekanan darah, ketebalan kulit, jumlah insulin, BMI, fungsi silsilah diabetes, dan umur. Dari proyek
ini diharapkan agar model dapat membantu perempuan untuk mengetahui kondisinya sedini mungkin.

**Daftar Referensi**

1. Rosita, R., Kusumaningtiar, D. A., Irfandi, A., & Ayu, I. M. (2022). Hubungan antara jenis kelamin, umur, dan
   aktivitas
   fisik dengan Diabetes melitus tipe 2 pada lansia di Puskesmas Balaraja Kabupaten Tangerang. Jurnal Kesehatan
   Masyarakat,
   10(3), 364-371.
2. Kementerian Kesehatan RI. Hari Diabetes
   Sedunia Tahun 2018. Pusat Data dan Informasi
   Kementrian Kesehatan RI. 2019; 1–8.

## Business Understanding

### Problem Statements

- Bagaimana membangun model machine learning mampu memprediksi apakah seorang wanita berisiko diabetes atau
  tidak?
- Bagaimana cara untuk mengatasi ketidakseimbangan kelas pada dataset untuk meningkatkan kinerja model?

### Goals

- Membuat model machine learning yang dapat memprediksi risiko diabetes pada wanita berdasarkan kondisi-kondisi tubuh
  yang ada
- Menggunakan teknik oversampling (SMOTE) untuk menangani kelas mintoritas (non-diabetes)

### Solution statements

- Membangun model dengan menggunakan tiga algoritma berbeda
- Melakukan hyperparameter tuning untuk meningkatkan kinerja dari ketiga model

## Data Understanding

Data yang digunakan pada proyek prediksi analitik ini adalah Diabetes dataset yang diunduh
dari [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). Dataset ini memiliki 768 data
dengan jumlah pembagian kelas sebanyak 500 adalah wanita non-diabetes dan 268 wanita mengidap diabetes. Dataset ini
memiliki ketidakseimbangan dalam jumlah kelas. Oleh karena itu, perlu dilakukan oversampling untuk mengatasi masalah
tersebut.

Dataset ini mencatat berbagai variabel fisiologis yang digunakan untuk menilai risiko diabetes, antara lain jumlah
kehamilan, kadar glukosa darah, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (BMI), fungsi silsilah
diabetes, dan usia.

### Variabel-variabel pada Diabetes Kaggle dataset adalah sebagai berikut:

- Pregnancies: merupakan jumlah kehamilan
- Glucose: merupakan kadar gula dalam darah setelah 2 jam melakukan Tes Toleransi Glukosa Oral (TTGO)
- BloodPressure: merupakan tekanan darah diastolic (mm Hg)
- SkinThickness: merupakan ketebalan lipatan kulit trisep (mm)
- Insulin: merupakan kadar insulin 2 jam setelah Tes Toleransi Glukosa (mm U/ml)
- BMI: merupakan indeks masa tubuh (berat dalam kg/(tinggi dalam m)^2)
- DiabetesPedigreeFunction: merupakan fungsi yang menghasilkan nilai pengaruh riwayat penyakit diabetes pada seseorang
- Age: merupakan umur (tahun)
- Outcome: merupakan kelas 0 (non-diabetes) atau 1 (diabetes)

### Data Visualization and EDA

**Melihat informasi dataset**

Berikut ini adalah informasi singkat mengenai dataset diabetes.
<img alt="Dataset Informartion" src="/assets/dataset-info.png" width="100%"/>
Terlihat pada gambar di atas terdapat 768 data dan 9 kolom/fitur pada dataset. Tidak ditemukan adanya missing value pada
masing-masing fitur. Untuk kolom Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Age, dan Outcome memiliki
tipe data int64. Kemudian, untuk kolom BMI dan DiabetesPedigreeFunction memiliki tipa data float64.

**Melihat deskripsi statistik pada dataset**

Berikut ini adalah deskripsi statistik dari dataset.

|       | Pregnancies | Glucose    | BloodPresure | SkinThickness | Insulin    | BMI        | DiabetesPedigreeFunction | Age        | Outcome    |
|-------|-------------|------------|--------------|---------------|------------|------------|--------------------------|------------|------------|
| count | 768.000000  | 768.000000 | 768.000000   | 768.000000    | 768.000000 | 768.000000 | 768.000000               | 768.000000 | 768.000000 |
| mean  | 3.845052    | 120.894531 | 69.105469    | 20.536458     | 79.799479  | 31.992578  | 0.471876                 | 33.240885  | 0.348958   |
| std   | 3.369578    | 31.972618  | 19.355807    | 15.952218     | 115.244002 | 7.884160   | 0.331329                 | 11.760232  | 0.476951   |
| min   | 0.000000    | 0.000000   | 0.000000     | 0.000000      | 0.000000   | 0.000000   | 0.078000                 | 21.000000  | 0.000000   |
| 25%   | 1.000000    | 99.000000  | 62.000000    | 0.000000      | 0.000000   | 27.300000  | 0.243750                 | 24.000000  | 0.000000   |
| 50%   | 3.000000    | 117.000000 | 72.000000    | 23.000000     | 30.500000  | 32.000000  | 0.372500                 | 29.000000  | 0.000000   |
| 75%   | 6.000000    | 140.250000 | 80.000000    | 32.000000     | 127.250000 | 36.600000  | 0.626250                 | 41.000000  | 1.000000   |
| max   | 17.000000   | 199.000000 | 122.000000   | 99.000000     | 846.000000 | 67.100000  | 2.420000                 | 81.000000  | 1.000000   |

Dari tabel di atas terlihat jumlah, rata-rata, standard deviasi, nilai minimum dan maksimum, dan kuartil 1 - 3 untuk
masing-masing fitur numerik. Pada tabel terlihat untuk kolom Glucose, BloodPresure, SkinThickness, Insulin, dan BMI
untuk nilai minimal adalah 0. Hal ini sangat tidak mungkin karena nilai tersebut untuk setiap orang pasti memilikinya.
Oleh
karena itu, permasalahan missing value ini perlu ditangani.

**Menangani Missing Value**

Dari hasil fungsi describe() untuk kolom Glucose, BloodPresure, SkinThickness, Insulin, dan BMI nilai minimumnya
adalah 0. Kondisi tersebut tidak mungkin dan dapat dianggap sebagai missing value. Berikut ini adalah potongan kode yang
digunakan untuk menghitung jumlah data yang bernilai 0.

```python
miss_val_count = (diabetes_df[col] == 0).sum()
```

Gambar di bawah ini adalah jumlah data yang terdapat 0 pada masing-masing kolom.

<img alt="Missing Value Count" src="/assets/missing-value-count.png" width=100%/>
Pada kolom Glocose terdapat 5 data yang terdapat nilai 0, BloodPressure ada 35 data, SkinThickness ada 227 data, Insulin
ada 374 data, dan kolom BMi ada 11 data. Untuk menangani masalah missing value digunakan metode imputasi regresi.

```python
IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
```

Kode di atas adalah penggunaan regresi dengan estimator yaitu RandomForestRegressor untuk mengatasi missing value.
Imputasi regresi bekerja dengan cara memperhitungkan hubungan antara variabel dan mempertahankan distribusi asli. Hal
ini dapat mengurangi bias dan meningkatkan akurasi dalam mengimputasikan nilai yang hilang.

**Melihat Distribusi Data pada Tiap Kolom**

Menggunakan fungsi hist() untuk melihat distribusi data dari masing-masing fitur.
<img alt="Missing Value Count" src="/assets/column-distribution.png" width=100%/>
Dari hasil distribusi data di atas, terlihat bahwa semua fitur memiliki distribusi cenderung right-skewed. Sebagian
besar populasi data terkonsentrasi pada bagian kiri. Pada distribusi data ini, nilai mean/rata-rata lebih besar dari
nilai median dan juga modus.

**Melihat Korelasi Fitur**

Menggunakan fungsi corr() untuk mendapatkan nilai korelasi antar fitur dan melakukan plot dengan heatmap().
<img alt="Correlation Matrix" src="/assets/matrix-correlation.png" width=100% />
Gambar di atas adalah heatmap yang menampilkan nilai korelasi antar fitur. Jika 

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada
notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu
menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan
  hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan
  mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek
berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**.
Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang
diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen
  markdown di situs
  editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/),
  atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan
  keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.