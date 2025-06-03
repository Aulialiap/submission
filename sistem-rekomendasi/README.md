# Laporan Proyek Machine Learning - Aulia Putri Fanani

## Project Overview

Sistem rekomendasi buku menjadi solusi penting untuk membantu pengguna menavigasi informasi yang melimpah dan meningkatkan pengalaman membaca[[1]](https://www.unesco.org/en/literacy). 
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
Dataset yang digunakan dalam proyek ini diambil dari platform Kaggle oleh [Arashnic](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/code). Dataset ini terdiri dari tiga file utama dalam format .csv, yaitu Books.csv, Users.csv, dan Ratings.csv, yang berisi informasi terkait buku, pengguna, dan interaksi berupa rating. Tujuan dari dataset ini adalah untuk membangun sistem rekomendasi buku berdasarkan preferensi pengguna.
Rincian data:
- Books.csv berisi 271,360 entri buku dengan fitur seperti ISBN, judul buku, penulis, tahun terbit, dan penerbit.
- Users.csv mencakup 278,858 data pengguna yang terdiri dari User-ID, lokasi, dan usia (jika tersedia).
- Ratings.csv memuat 1.149.780 entri interaksi berupa rating antara pengguna dan buku, dengan nilai rating antara 0 hingga 10.
Secara umum, kondisi data cukup lengkap namun terdapat beberapa missing values dan duplikasi yang perlu dibersihkan dan hanya akan digunakan untuk training sekitar 10.000 sample teratas
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
Pada tahap ini, dilakukan beberapa langkah data preparation untuk memastikan kualitas data yang baik sebelum membangun sistem rekomendasi. Yang dilakukan diantaranya 
### **Menggabungkan Dataset**
Dataset Ratings.csv digabungkan dengan Books.csv dan Users.csv berdasarkan kolom ISBN dan User-ID untuk membentuk satu dataset terpadu yang berisi informasi pengguna, buku, dan rating. Penggabungan ini penting untuk memungkinkan analisis menyeluruh dan membangun sistem rekomendasi berbasis interaksi. Sehingga didapat 
  - Jumlah seluruh data buku berdasarkan ISBN : 341765
  - Jumlah seluruh user berdasarkan User-ID : 278858
  - Jumlah total rating yang tersedia: 1149780
  - Jumlah user  yang memberi rating: 105283
  - Jumlah buku  yang diberi rating: 340556

### **Menghapus Nilai Rating Nol**
Rating dengan nilai 0 dianggap sebagai implicit feedback (pengguna telah melihat buku tetapi tidak memberikan penilaian eksplisit) dan terdapat indikasi missing value karena kesenjangan jumlah yang begitu besar. Karena proyek ini berfokus pada sistem rekomendasi berbasis explicit feedback, maka entri dengan rating 0 dihapus untuk meningkatkan keakuratan model.

### **Menghapus Duplikasi**
Diperiksa adanya data duplikat pada hasil penggabungan dataset dan akan dihapus untuk menghindari bias dalam model dan evaluasi.
![image](https://github.com/user-attachments/assets/f416bdcc-5886-4d81-a01d-27417d12e945)

### **Menangani Missing Values**
Data pada kolom Age ditemukan memiliki nilai kosong. Nilai yang tidak valid dihapus atau diisi dengan median untuk menjaga distribusi data tetap wajar. Seperti pada variabel Age yang menerapkan pengisian menggunakan median usia pengguna nya. Ada kemungkinan dalam data hasil merge (ratings + books), sebagian ISBN tidak ditemukan datanya di books.csv, sehingga seluruh informasi tentang buku (judul, penulis, tahun, penerbit) jadi kosong semua. Sehingga semua kolom informasi buku kosong, maka baris itu tidak bisa digunakan untuk sistem rekomendasi (tidak ada data konten) dan Usia pengguna ditangani dengan median agar tidak bias.
all_book.dropna(subset=['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'], inplace=True)
all_book['Age'] = all_book['Age'].fillna(all_book['Age'].median())
all_book.drop(all_book[all_book["Book-Rating"] == 0].index, inplace=True)

![image](https://github.com/user-attachments/assets/6ddb8062-52fd-4577-b010-652fc13b78eb)

### **Mengonversi kolom DataFrame menjadi list python**
Mengonversi kolom DataFrame menjadi list Python, supaya bisa diproses lebih fleksibel (misalnya untuk model, dictionary, looping) dan dapat digunakan untuk TF-IDF vectorization, cosine similarity, dan fungsi rekomendasi menggunakan fungsi tolist().

## Modeling
Pada tahap modeling, sistem rekomendasi dikembangkan menggunakan metode Content Based Filtering yang mengandalkan fitur konten buku, yaitu penulis (author), untuk menentukan kemiripan antar buku.

Kelebihan:
- Dapat memberikan rekomendasi yang dipersonalisasi berdasarkan preferensi pengguna lain dengan pola serupa.
- Tidak memerlukan informasi tambahan tentang buku (judul, genre, dll).

Kekurangan:
- Tidak dapat memberikan rekomendasi kepada pengguna atau buku baru yang belum memiliki cukup data interaksi (cold start problem).

Prosesnya meliputi
#### 1. Feature Extraction : 
Nama penulis buku diubah menjadi representasi numerik menggunakan TF-IDF Vectorizer untuk menangkap bobot kata-kata penting dalam nama penulis. Agar sistem dapat memahami dan membandingkan konten buku, kita perlu mengubah teks menjadi bentuk numerik. Salah satu teknik yang umum digunakan adalah TF-IDF (Term Frequency-Inverse Document Frequency) dan akan diambil sebanyak 10.000 buku terpopuler
  
#### 2. Cosine Similarity : 
Fungsi ini menghitung kemiripan antar setiap pasangan buku berdasarkan vektor TF-IDF yang telah dihasilkan sebelumnya. Matriks cosine similarity ini kemudian disimpan dalam bentuk DataFrame, dengan baris dan kolom diberi label nama judul buku. Struktur ini memudahkan proses pencarian dan interpretasi ketika sistem ingin merekomendasikan buku yang mirip dengan buku tertentu.
  
#### 3. Rekomendasi Top-N : 
Fungsi ini digunakan untuk memberikan rekomendasi buku berdasarkan kemiripan penulis, dengan memanfaatkan matriks cosine similarity antar buku. Pengguna memasukkan judul buku sebagai acuan, lalu fungsi mencari buku-buku lain yang paling mirip berdasarkan nilai similarity tersebut.
Parameter yang digunakan :  
- **nama_buku** (`str`): Judul buku sebagai acuan rekomendasi.  
- **similarity_data** (`pd.DataFrame`): Matriks cosine similarity antar buku.  
- **items** (`pd.DataFrame`): DataFrame yang memuat kolom `'book_title'` dan `'author'`.  
- **k** (`int`): Jumlah rekomendasi yang ingin dihasilkan.

Misalnya cari rekomendasi berdasarkan salah satu judul : "Harry Potter and the Order of the Phoenix (Book 5)"
![image](https://github.com/user-attachments/assets/f2bd37e7-997e-46fc-9b53-e8e3e57d6be5)

Akan memunculkan rekomendasi buku yang mirip dengan yang dicari

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah Root Mean Squared Error (RMSE). RMSE adalah salah satu metrik yang paling umum digunakan dalam regresi untuk mengukur seberapa jauh prediksi model dari nilai aktual.

![image](https://github.com/user-attachments/assets/206d7305-2805-4b72-aa80-68bdfd54bd48)

Dimana yi adalah nilai aktual, dan 𝑦^𝑖 adalah nilai prediksi.
Berdasarkan grafik hasil pelatihan model, terlihat bahwa nilai RMSE pada data training menurun drastis dalam beberapa epoch pertama, lalu stabil di sekitar angka 0.030. Sedangkan pada data testing, nilai RMSE juga cukup stabil di sekitar 0.035, meskipun sedikit lebih tinggi dari data training. Perbedaan ini menunjukkan bahwa model belajar dengan baik dari data training dan tidak mengalami overfitting secara signifikan. Sehingga ketika di test model dapat menghasilkan 10 rekomendasi buku teratas untuk satu pengguna yang dipilih secara acak.

![image](https://github.com/user-attachments/assets/b7a1d990-fcc3-4a78-8e9f-7bd51f9c6f25)


