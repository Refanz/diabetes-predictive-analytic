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
untuk terkena diabetes melitus tipe 2 dibandingkan dengan laki-laki.<sup>1</sup> Kemudian, berdasarkan riset kesehatan dasar
RISKESDAS 2018 penderita diabetes melitius pada perempuan (1,8%) lebih tinggi daripada laki-laki (1,2%) di Indonesia.

Risiko terkena diabetes pada perempuan lebih tinggi daripada laki-laki. Oleh karena itu, proyek ini fokus pada
pengembangan
model machine learning untuk memprediksi risiko perempuan terkena diabetes atau tidak. Terdapat 768 data
dengan pembagian sebanyak 500 tidak mengidap diabetes dan 268 mengalami diabetes. Kemudian untuk variabel prediksi
adalah jumlah
kehamilan, glukosa, tekanan darah, ketebalan kulit, jumlah insulin, BMI, fungsi silsilah diabetes, dan umur. Dari proyek
ini diharapkan agar model dapat membantu perempuan untuk mengetahui kondisinya sedini mungkin.

**Daftar Pustaka**

1. Rosita, R., Kusumaningtiar, D. A., Irfandi, A., & Ayu, I. M. (2022). Hubungan antara jenis kelamin, umur, dan
   aktivitas
   fisik dengan Diabetes melitus tipe 2 pada lansia di Puskesmas Balaraja Kabupaten Tangerang. Jurnal Kesehatan
   Masyarakat,
   10(3), 364-371.
2. Kementerian Kesehatan RI. Hari Diabetes
   Sedunia Tahun 2018. Pusat Data dan Informasi
   Kementrian Kesehatan RI. 2019; 1–8.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:

- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang
diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan
  sebagai berikut:

  ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi
      yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau
tautan untuk mengunduh dataset.
Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:

- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory
  data analysis.

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