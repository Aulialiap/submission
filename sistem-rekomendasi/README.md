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

Terlihat terdapat beberapa missing value pada data awal pada dataset book.csv

![image](https://github.com/user-attachments/assets/344a8e48-d2ac-4f49-9603-9f1631a49945)


### Users.csv
![image](https://github.com/user-attachments/assets/acdece54-1ec0-4887-8380-256daf5608a2)
- User-ID : berisi ID unik pengguna
- Location : berisi data lokasi pengguna
- Age : berisi data usia pengguna

Terlihat ada missing value pada data awal pada dataset users.csv dari persebaran dataset yang tidak merata 

![image](https://github.com/user-attachments/assets/265bb5f7-9a54-4440-91ae-41439976830c)


### Rating.csv
![image](https://github.com/user-attachments/assets/e6d7dee9-3dba-4b84-8a61-fdcfaf28b40b)
- User-ID : berisi ID unik pengguna
- ISBN : berisi kode ISBN buku yang diberi rating oleh pengguna
- Book-Rating : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

Terlihat tidak ada missing value pada data awal pada dataset rating.csv

![image](https://github.com/user-attachments/assets/c177153a-dcac-4660-9118-459b6282690b)

Namun pada distribusi data rating.csv terdapat nilai rating 0 yang berarti rating implisit (pengguna telah melihat atau memiliki buku tapi tidak memberi rating eksplisit) sehingga perlu dibersihkan untuk menghindari bias

![image](https://github.com/user-attachments/assets/2059a6ba-50b7-46c4-99f5-150e81d3fa17)

Pada ketiga dataset tersebut juga tidak terlihat adanya duplikasi data yang berarti dataset sudah siap untuk dianalisis lebih lanjut.

![image](https://github.com/user-attachments/assets/6f39abb5-0885-4b0c-ad16-7b0a5cc1e0d9)

## Data Preparation
Pada tahap ini, dilakukan beberapa langkah data preparation untuk memastikan kualitas data yang baik sebelum membangun sistem rekomendasi. Yang dilakukan diantaranya 

### **Menggabungkan Dataset**
Dataset Ratings.csv digabungkan dengan Books.csv dan Users.csv berdasarkan kolom ISBN dan User-ID untuk membentuk satu dataset terpadu yang berisi informasi pengguna, buku, dan rating. Penggabungan ini penting untuk memungkinkan analisis menyeluruh dan membangun sistem rekomendasi berbasis interaksi. Sehingga didapat 
  - Jumlah seluruh data buku berdasarkan ISBN : 341765
  - Jumlah seluruh user berdasarkan User-ID : 278858
  - Jumlah total rating yang tersedia: 1149780
  - Jumlah user  yang memberi rating: 105283
  - Jumlah buku  yang diberi rating: 340556

![image](https://github.com/user-attachments/assets/734b5a4d-5925-4451-868a-de3e4916a762)

Dengan menggabungkan ketiga file ini, kita mendapatkan satu DataFrame besar (all_book) yang menjadi sumber data utama untuk sistem rekomendasi dan menghasilkan satu dataset terpadu yang berisi informasi pengguna, buku, dan rating—sehingga analisis dan pemodelan rekomendasi dapat memanfaatkan ketiganya secara bersamaan.

### **Menangani Missing Values**
Menghapus dan membersihkan beberapa missing value pada fitur yang akan digunakan. Menghapus dan drop beberapa missing value dan pada variabel age yang diisi dengan median agar tidak bias.

![image](https://github.com/user-attachments/assets/263fe48f-a8c2-4a35-bb04-8caf542061d5)

![image](https://github.com/user-attachments/assets/6ddb8062-52fd-4577-b010-652fc13b78eb)

### **Menghapus Nilai Rating Nol**
Rating dengan nilai 0 dianggap sebagai implicit feedback (pengguna telah melihat buku tetapi tidak memberikan penilaian eksplisit) dan terdapat indikasi missing value karena kesenjangan jumlah yang begitu besar. Karena proyek ini berfokus pada sistem rekomendasi berbasis explicit feedback, maka entri dengan rating 0 dihapus untuk meningkatkan keakuratan model.

### **Menghapus Duplikasi**
Pada tahap ini, dibuat salinan dari dataset utama all_book ke dalam dataframe bernama preparation. Kemudian, dilakukan penghapusan duplikasi berdasarkan kolom ISBN dengan perintah :

![image](https://github.com/user-attachments/assets/f416bdcc-5886-4d81-a01d-27417d12e945)

Langkah ini bukan merupakan penghapusan duplikasi umum di seluruh dataset, melainkan bertujuan untuk memastikan bahwa setiap buku hanya direpresentasikan satu kali saat dilakukan proses TF-IDF dan perhitungan cosine similarity. Jika satu ISBN muncul lebih dari satu kali, maka dapat menyebabkan bias dalam representasi konten dan penghitungan kemiripan antar buku.


### **TF-IDF Vectorization**
Untuk sistem rekomendasi berbasis konten, kolom author, book-titile, publisher yang digabungkan dalam satu dataframe baru bernama content dan dikonversi menjadi vektor numerik menggunakan teknik TF-IDF agar dapat dibandingkan menggunakan cosine similarity.

![image](https://github.com/user-attachments/assets/04712c91-4c49-47ba-a745-9668ce1ab718)

Sebelum melakukan proses vektorisasi, nilai kosong (NaN) pada kolom penulis diisi terlebih dahulu dengan string kosong agar tidak menyebabkan error. Kemudian, TfidfVectorizer dari scikit-learn diinisialisasi. Setelah itu, dilakukan proses fit_transform pada kolom penulis untuk menghasilkan matriks TF-IDF, yaitu matriks berdimensi (jumlah buku × jumlah kata unik). Hasil inilah yang akan digunakan untuk menghitung kemiripan antar buku dengan cosine similarity. Fungsi ini menghitung kemiripan antar setiap pasangan buku berdasarkan vektor TF-IDF yang telah dihasilkan sebelumnya. Matriks cosine similarity ini kemudian disimpan dalam bentuk DataFrame, dengan baris dan kolom diberi label nama judul buku. Struktur ini memudahkan proses pencarian dan interpretasi ketika sistem ingin merekomendasikan buku yang mirip dengan buku tertentu.

### **Encoding dan Splitting dataset**
- Encoding kolom user dan book dilakukan encoding pada kolom User-ID dan ISBN menjadi indeks numerik (user dan book). Pada tahap ini, dilakukan pemetaan terhadap dataset utama agar setiap baris memiliki nilai numerik yang dapat digunakan oleh model pembelajaran mesin. 

![image](https://github.com/user-attachments/assets/73e632f1-a829-4284-9488-43c20bd5c7ca)

- Untuk mendukung stabilitas pelatihan model, nilai rating dinormalisasi ke dalam rentang 0 hingga 1 dengan membagi nilai rating asli (1–10) dengan 10:

![image](https://github.com/user-attachments/assets/e68c2e8d-5392-4089-a64d-d76ed6911587)

- Dataset kemudian dibagi menjadi dua subset: data pelatihan dan data validasi menggunakan rasio 80:20 untuk menguji performa model pada data yang belum pernah dilihat.

![image](https://github.com/user-attachments/assets/77f2e55d-37dd-40ba-8b7c-d54441c4ee93)

Selain itu, dilakukan penyaringan pada dataset all_book untuk hanya menyertakan buku-buku populer yang dianggap relevan. Hal ini bertujuan untuk mengurangi noise dari buku-buku yang sangat jarang diberi rating, yang dapat menyebabkan data menjadi terlalu sparse dan menurunkan performa model. Penyaringan ini dilakukan dengan cara :
- Menghitung jumlah total rating untuk setiap judul buku.
- Menyaring hanya buku dengan jumlah rating di atas ambang batas tertentu sebagai top_books
- Dataset all_book kemudian difilter menggunakan :

![image](https://github.com/user-attachments/assets/f1527f95-a3b6-4ac9-bd54-394af6bff5f7)

Dengan hanya mempertahankan top_books, model dapat fokus pada item yang memiliki informasi interaksi cukup kaya untuk dipelajari.

## Modeling and Result
Pada tahap modeling, sistem rekomendasi dikembangkan menggunakan pendekatan utama content based filtering. 

Kelebihan:
- Dapat memberikan rekomendasi yang dipersonalisasi berdasarkan preferensi pengguna lain dengan pola serupa.
- Tidak memerlukan informasi tambahan tentang buku (judul, genre, dll).

Kekurangan:
- Tidak dapat memberikan rekomendasi kepada pengguna atau buku baru yang belum memiliki cukup data interaksi (cold start problem).

Pendekatan ini merekomendasikan buku berdasarkan kemiripan kontennya. Sistem ini menggunakan penulis (author) sebagai fitur utama untuk menghitung kemiripan antar buku. Berikut langkah-langkahnya:

#### 1. Cosine Similarity
Kemiripan antar buku dihitung menggunakan cosine similarity terhadap vektor TF-IDF hasil ekstraksi fitur Book-Author. Matriks kemiripan ini membandingkan setiap buku dengan buku lainnya. Fungsi ini menghitung kemiripan antar setiap pasangan buku berdasarkan vektor TF-IDF yang telah dihasilkan sebelumnya. Matriks cosine similarity ini kemudian disimpan dalam bentuk DataFrame, dengan baris dan kolom diberi label nama judul buku. Struktur ini memudahkan proses pencarian dan interpretasi ketika sistem ingin merekomendasikan buku yang mirip dengan buku tertentu.

#### 2. Top-N Recommendation
Fungsi recommend_books_by_author() digunakan untuk mencari buku yang paling mirip berdasarkan nama penulis. Fungsi ini digunakan untuk memberikan rekomendasi buku berdasarkan kemiripan penulis, dengan memanfaatkan matriks cosine similarity antar buku. Pengguna memasukkan judul buku sebagai acuan, lalu fungsi mencari buku-buku lain yang paling mirip berdasarkan nilai similarity tersebut. Fungsi ini menerima parameter:
- **nama_buku** (`str`): Judul buku sebagai acuan rekomendasi.  
- **similarity_data** (`pd.DataFrame`): Matriks cosine similarity antar buku.  
- **items** (`pd.DataFrame`): DataFrame yang memuat kolom `'book_title'` dan `'author'`.  
- **k** (`int`): Jumlah rekomendasi yang ingin dihasilkan.

**Contoh Hasilnya** adalah ketika mencari rekomendasi dari buku Harry Potter and the Order of the Phoenix (Book 5) maka akan muncul beberapa rekomendasi buku yang mirip dan mungkin akan menarik minat pembaca

![image](https://github.com/user-attachments/assets/02f99b15-26a8-40b3-a7e0-5c68fe993094)

Akan memunculkan rekomendasi buku yang mirip dengan yang dicari

Untuk pendekatan yang lebih adaptif terhadap perilaku pengguna, dibangun juga model deep learning berbasis Collaborative Filtering menggunakan arsitektur RecommenderNet. Model ini menggunakan pasangan (user, book) dan mempelajari pola rating yang diberikan dengan arsitektur model : 

![image](https://github.com/user-attachments/assets/96ded686-2ef1-4973-a56f-760721ed8fbc)

Model RecommenderNet dibangun menggunakan TensorFlow Keras dengan class kustom tf.keras.Model. Berikut adalah komponen utama dari arsitekturnya:
a. User & Book Embeddings
- user_embedding: Mengubah ID pengguna menjadi vektor berdimensi embedding_size.
- book_embedding: Mengubah ID buku menjadi vektor berdimensi embedding_size.
- Fungsi: Memproyeksikan user dan item ke dalam ruang laten sehingga interaksi antar keduanya bisa dihitung melalui dot product.
b. Bias Embedding
- user_bias dan book_bias: Layer tambahan untuk memodelkan bias masing-masing pengguna dan buku terhadap nilai rating rata-rata.
c. Dot Product + Bias
- Interaksi antara pengguna dan buku dihitung melalui dot product antara vektor embedding, lalu ditambahkan dengan bias pengguna dan buku :

  ![image](https://github.com/user-attachments/assets/dd245123-8f80-4ccc-b18a-c1ca774245a4)

Model dikompilasi menggunakan :
- Loss Function: MeanSquaredError() - digunakan karena ini adalah regresi untuk memprediksi nilai rating.
- Optimizer: Adam - digunakan karena kemampuannya beradaptasi terhadap learning rate dan stabil untuk pelatihan embedding.

![image](https://github.com/user-attachments/assets/6c9c57e7-a3de-4fc3-a22a-f0dcc4436463)

Model dilatih menggunakan :
- Epoch: 10 - menentukan berapa kali seluruh dataset dilalui oleh model.
- Batch Size: 64 - banyaknya sampel dalam satu langkah update.
- Validation Split: 20% dari data digunakan untuk validasi agar tidak overfitting.

Setelah pelatihan, model dapat digunakan untuk memprediksi nilai rating untuk pasangan (user, book) yang belum pernah dinilai, lalu memilih buku dengan rating prediksi tertinggi sebagai rekomendasi untuk pengguna tersebut.

![image](https://github.com/user-attachments/assets/b7a1d990-fcc3-4a78-8e9f-7bd51f9c6f25)

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah Root Mean Squared Error (RMSE). RMSE adalah salah satu metrik yang paling umum digunakan dalam regresi untuk mengukur seberapa jauh prediksi model dari nilai aktual.

![image](https://github.com/user-attachments/assets/206d7305-2805-4b72-aa80-68bdfd54bd48)

Dimana yi adalah nilai aktual, dan 𝑦^𝑖 adalah nilai prediksi.
Berdasarkan grafik hasil pelatihan model, terlihat bahwa nilai RMSE pada data training menurun drastis dalam beberapa epoch pertama, lalu stabil di sekitar angka 0.030. Sedangkan pada data testing, nilai RMSE juga cukup stabil di sekitar 0.035, meskipun sedikit lebih tinggi dari data training. Perbedaan ini menunjukkan bahwa model belajar dengan baik dari data training dan tidak mengalami overfitting secara signifikan. 
Berdasarkan hasil evaluasi yang telah dilakukan, sistem terbukti memberikan rekomendasi yang akurat, baik dari sisi prediksi rating (dengan metrik RMSE) 

![image](https://github.com/user-attachments/assets/dbe267b4-6497-4bfd-ad78-d236c272eb95)

Untuk mengevaluasi model Content-Based Filtering, digunakan metrik evaluasi berbasis ranking yang umum digunakan pada sistem rekomendasi, yaitu:
- Precision@K
- Recall@K
- F1-Score@K
- NDCG@K (Normalized Discounted Cumulative Gain)

Evaluasi dilakukan terhadap 20 pengguna yang dipilih secara acak. Sistem memberikan Top-5 rekomendasi untuk masing-masing pengguna berdasarkan kemiripan konten (gabungan judul, penulis, dan penerbit dengan hasil berikut

![image](https://github.com/user-attachments/assets/d90a15ed-c87b-4565-a808-f5b8e0688dd5)

Hasil tersebut menunjukkan bahwa sistem Content-Based Filtering cukup baik dalam memberikan rekomendasi yang relevan dan memprioritaskan buku-buku yang sesuai preferensi pengguna dalam urutan atas.

Sistem secara langsung menanggapi pernyataan masalah utama, yaitu kesulitan pengguna dalam menemukan buku yang sesuai dengan preferensi mereka. Dengan adanya dua pendekatan—Content-Based dan Collaborative Filtering—pengguna dapat menerima rekomendasi yang lebih relevan baik berdasarkan kemiripan konten maupun berdasarkan pola perilaku pengguna lain.

Tujuan utama proyek ini adalah memberikan rekomendasi yang personal dan tepat sasaran. Evaluasi menggunakan RMSE menunjukkan bahwa model memiliki performa prediksi yang baik (nilai error rendah), dan metrik evaluasi Content-Based Filtering menunjukkan bahwa sistem memberikan hasil yang relevan dan berkualitas. Maka dapat disimpulkan bahwa sistem telah mencapai goal yang ditetapkan di awal. Solusi yang diterapkan—yaitu penerapan model deep learning untuk collaborative filtering dan pemanfaatan TF-IDF untuk content based filtering mampu meningkatkan kualitas rekomendasi. Dampaknya tidak hanya terlihat dari hasil evaluasi metrik, tetapi juga dari fleksibilitas sistem dalam memberikan rekomendasi kepada pengguna dengan atau tanpa riwayat interaksi.


