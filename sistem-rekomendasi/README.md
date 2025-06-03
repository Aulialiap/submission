# Laporan Proyek Machine Learning - Aulia Putri Fanani

## Project Overview

Sistem rekomendasi buku menjadi solusi penting untuk membantu pengguna menavigasi informasi yang melimpah dan meningkatkan pengalaman membaca. 
Dengan menggabungkan metode content-based filtering (berdasarkan kemiripan konten buku), proyek ini membangun sistem rekomendasi 
yang mampu menyarankan buku secara lebih personal dan relevan. Proyek ini menggunakan dataset publik dari Kaggle yang mencakup data buku, pengguna, dan rating. 
Dengan pendekatan berbasis machine learning dan deep learning, sistem ini tidak hanya membantu pengguna menemukan buku yang sesuai, 
tetapi juga memberikan nilai tambah bagi platform digital, toko buku online, atau perpustakaan dalam meningkatkan keterlibatan pengguna.

## Business Understanding
### Problem Statements
- Banyak pengguna kesulitan menemukan buku yang relevan dengan minat mereka karena jumlah pilihan yang sangat banyak.
- Sistem rekomendasi yang ada terkadang tidak personal

### Goals
- Mengembangkan sistem rekomendasi buku yang dapat menyarankan bacaan sesuai minat dan preferensi pengguna secara personal.
- Meningkatkan akurasi dan relevansi rekomendasi untuk pembaca

### Solution Approach 
- Content-Based Filtering: Merekomendasikan buku berdasarkan kesamaan atribut konten seperti judul, penulis, dan kategori, yang mirip dengan buku yang disukai pengguna sebelumnya.

## Data Understanding
Dataset ini terdiri dari tiga file utama, yaitu Books.csv, Users.csv, dan Ratings.csv. Total data terdiri dari lebih dari 1 juta entri rating buku oleh pengguna, 
mencakup ribuan judul buku dan pengguna yang berbeda. Secara umum, kondisi data cukup lengkap namun terdapat beberapa missing values dan duplikasi yang perlu dibersihkan.
Berikut deskripsi masing-masing fitur pada dataset :

### Book.csv
![image](https://github.com/user-attachments/assets/ef16ea98-239f-449d-a792-ce96b3b0a9d5)
- ISBN : Nomor unik identifikasi buku.
- Book-Title : Judul buku.
- Book-Author : Nama penulis.
- Year-Of-Publication : Tahun terbit buku.
- Publisher : Nama penerbit buku.
- Image-URL-S, Image-URL-M, Image-URL-L : URL gambar sampul buku dalam tiga ukuran.

### Users.csv
![image](https://github.com/user-attachments/assets/acdece54-1ec0-4887-8380-256daf5608a2)
- User-ID : berisi ID unik pengguna
- Location : berisi data lokasi pengguna
- Age : berisi data usia pengguna

### Rating.csv
![image](https://github.com/user-attachments/assets/e6d7dee9-3dba-4b84-8a61-fdcfaf28b40b)
- User-ID : berisi ID unik pengguna
- ISBN : berisi kode ISBN buku yang diberi rating oleh pengguna
- Book-Rating : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
