# Laporan Proyek Machine Learning - Aulia Putri Fanani

## Domain Proyek

Penyakit diabetes merupakan salah satu penyakit kronis yang paling umum dan berbahaya secara global. Menurut WHO, jumlah penderita diabetes terus meningkat setiap tahun dan menjadi penyebab utama komplikasi serius seperti gagal ginjal dan penyakit jantung. Oleh karena itu, deteksi dini sangat penting untuk mencegah dampak jangka panjang [[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes)

Dengan memanfaatkan predictive analytics berbasis machine learning, data kesehatan seperti kadar glukosa, tekanan darah, BMI, dan riwayat keluarga dapat dianalisis untuk memprediksi kemungkinan seseorang menderita diabetes. Proyek ini membandingkan tiga model regresi yaitu K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost untuk menemukan model terbaik dalam memprediksi risiko diabetes menggunakan dataset dari Kaggle (https://www.kaggle.com/datasets/mathchi/diabetes-data-set).

Pendekatan ini bertujuan membantu tenaga medis atau sistem kesehatan dalam pengambilan keputusan berbasis data, sehingga penanganan pasien berisiko tinggi dapat dilakukan lebih cepat dan akurat.

## Business Understanding
### Problem Statements
- Bagaimana membangun model machine learning untuk memprediksi risiko diabetes berdasarkan data kesehatan pasien?
- Model machine learning mana yang memberikan performa terbaik dalam memprediksi diabetes?
- Bagaimana model prediksi ini dapat membantu tenaga medis dalam deteksi dini dan pengambilan keputusan?

### Goals
- Membangun model prediksi diabetes berdasarkan fitur-fitur seperti kadar glukosa, tekanan darah, BMI, dan lain-lain.
- Membandingkan performa model K-Nearest Neighbors, Random Forest, dan AdaBoost untuk menemukan yang paling akurat.
- Memberikan dukungan berbasis data kepada tenaga medis untuk melakukan deteksi dini terhadap risiko diabetes.

### Solution statements
- Melakukan eksplorasi dan pembersihan data, termasuk menangani nilai yang hilang dan melakukan normalisasi fitur.
- Mengimplementasikan dan membandingkan tiga model machine learning: KNN, Random Forest, dan AdaBoost.
- Mengevaluasi kinerja model menggunakan metrik seperti accuracy, precision, recall, dan F1-score untuk menentukan model terbaik.

## Data Understanding
### Deskripsi Variabel
Dataset yang digunakan dalam proyek ini diambil dari platform Kaggle dan dapat diakses melalui tautan berikut: Diabetes Data Set by [Mathchi](https://www.kaggle.com/datasets/mathchi/diabetes-data-set), yang dipublikasikan oleh [Mathchi](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). Dataset ini terdiri dari 768 data pasien dan 9 fitur, yang mencakup informasi medis seperti kadar glukosa, tekanan darah, BMI, dan riwayat kehamilan, yang bertujuan untuk memprediksi kemungkinan seseorang terkena diabetes. Fitur-fitur dalam dataset ini seluruhnya bertipe numerik dan tidak memiliki missing value ataupun data duplikat. Dataset disimpan dalam format .csv dan hanya terdiri dari satu file utama.

Adapun deskripsi dari masing-masing kolom adalah sebagai berikut:
| No | Nama Kolom                   | Tipe Data | Deskripsi                                                           |
| -- | ---------------------------- | --------- | ------------------------------------------------------------------- |
| 1  | **Pregnancies**              | `int64`   | Jumlah kehamilan yang pernah dialami pasien                         |
| 2  | **Glucose**                  | `int64`   | Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral |
| 3  | **BloodPressure**            | `int64`   | Tekanan darah diastolik (dalam mm Hg)                               |
| 4  | **SkinThickness**            | `int64`   | Ketebalan lipatan kulit triceps (dalam mm)                          |
| 5  | **Insulin**                  | `int64`   | Kadar insulin serum 2 jam setelah makan (dalam μU/ml)               |
| 6  | **BMI**                      | `float64` | Indeks Massa Tubuh = berat (kg) / (tinggi (m))²                     |
| 7  | **DiabetesPedigreeFunction** | `float64` | Nilai fungsi silsilah keluarga terhadap risiko diabetes             |
| 8  | **Age**                      | `int64`   | Usia pasien (dalam tahun)                                           |
| 9  | **Outcome**                  | `int64`   | Variabel target (0 = tidak diabetes, 1 = diabetes)                  |

Pada tahap awal eksplorasi data, dilakukan analisis statistik deskriptif menggunakan fungsi describe() untuk memahami sebaran nilai pada setiap fitur numerik. Dari hasil ringkasan statistik tersebut, dapat diketahui rata-rata, standar deviasi, nilai minimum, maksimum, serta kuartil dari masing-masing variabel. Analisis ini membantu memberikan gambaran awal terhadap data, seperti nilai-nilai ekstrem atau ketidakseimbangan dalam skala antar fitur.

![image](https://github.com/user-attachments/assets/7f43f35a-d0c7-42d3-8f39-11156b5b6a6f)

Fungsi `describe()` digunakan untuk menampilkan statistik deskriptif dari setiap kolom numerik dalam dataset. Informasi yang diberikan meliputi:
- `Count`: Jumlah data (non-null) pada setiap kolom.
- `Mean`: Nilai rata-rata dari kolom.
- `Std`: Standar deviasi, mengukur sebaran data terhadap nilai rata-rata.
- `Min`: Nilai minimum dalam kolom.
- `25%`: Kuartil pertama (Q1), yaitu nilai yang memisahkan 25% data terbawah.
- `50%`: Kuartil kedua (Q2) atau median, yaitu nilai tengah dari data.
- `75%`: Kuartil ketiga (Q3), yaitu nilai yang memisahkan 25% data teratas.
- `Max`: Nilai maksimum dalam kolom.

### Menangani Missing Value dan Duplikasi Data
Pada tahap awal, dilakukan pengecekan missing value pada seluruh variabel menggunakan fungsi isna().sum(). Hasil pengecekan menunjukkan bahwa dataset ini bebas dari nilai kosong (missing value) pada semua kolom, sehingga tidak diperlukan penanganan khusus untuk data yang hilang. Selanjutnya, pengecekan duplikasi data juga dilakukan dengan fungsi duplicated().sum(), dan hasilnya menunjukkan tidak adanya baris data yang duplikat, yang berarti dataset sudah siap untuk dianalisis lebih lanjut.

![image](https://github.com/user-attachments/assets/a9fe0086-2001-4e95-944b-84eeaedb5cfd)

### Deteksi dan Penanganan Outlier
Kemudian dilakukan visualisasi menggunakan diagram boxplot untuk masing-masing fitur numerik. Dari hasil visualisasi tersebut, ditemukan adanya outlier terutama pada fitur seperti Insulin, SkinThickness, dan BMI. Outlier ini kemudian ditangani dengan metode Interquartile Range (IQR), yaitu dengan menghapus data yang berada di luar batas bawah dan atas berdasarkan distribusi kuartil. Pada proses ini, nilai kuartil pertama (Q1), kuartil ketiga (Q3), serta selisih antar kuartil (IQR) dihitung untuk tiap kolom numerik. Data yang memiliki nilai di luar rentang Q1 - 1.5 × IQR dan Q3 + 1.5 × IQR dianggap outlier dan kemudian dihapus. Setelah proses penyaringan, dataset bersih tersisa sebanyak 639 baris dan 9 kolom, yang siap untuk tahap analisis selanjutnya.

### Univariate Analysis
Analisis univariat dilakukan dengan menampilkan histogram untuk setiap fitur numerik, yang membantu memvisualisasikan distribusi data. Dari histogram ini, terlihat variasi distribusi antar fitur, yang memberikan gambaran karakteristik masing-masing variabel seperti skewness atau penyebaran nilai seperti berikut

![image](https://github.com/user-attachments/assets/ec1906b9-3a40-4894-9837-6f10bd4d6327)

### Multivariate Analysis
Selanjutnya, analisis multivariat dilakukan dengan menggunakan pairplot untuk mengamati hubungan antar fitur numerik sekaligus hubungan fitur dengan variabel target. Visualisasi ini memperlihatkan pola distribusi bersama dan potensi korelasi antara variabel-variabel dalam dataset. Untuk menguatkan hasil ini, dibuat pula correlation matrix dalam bentuk heatmap yang menampilkan nilai korelasi antar fitur numerik secara kuantitatif sebagai berikut 

![image](https://github.com/user-attachments/assets/4d9c6e39-7838-45ab-a0f2-765b15faece9)

![image](https://github.com/user-attachments/assets/f0df45af-86bf-4085-b6d4-d76a9d122781)

## Data Preparation
Tahapan bertujuan untuk menyiapkan data agar optimal untuk proses pelatihan model. Tahapan yang dilakukan meliputi :
- Reduksi dimensi dengan Principal Component Analysis (PCA).
Fitur yang dipilih setelah melihat korelasi terkuat dengan outcome adalah ['Glucose', 'BMI', 'Age', 'Outcome']

![image](https://github.com/user-attachments/assets/ecf2d8ec-c60d-4755-8a7b-fbc2d8526282)

- Pembagian dataset dengan fungsi train_test_split dari library sklearn.

  Setelah fitur dipilih, dataset kemudian dipisahkan menjadi fitur (X) dan target (y), lalu dilakukan pembagian dataset menjadi data latih (training set) dan data uji (test set). Pembagian dilakukan dengan proporsi 90% data latih dan 10% data uji menggunakan fungsi train_test_split dari pustaka Scikit-Learn. 
- Standarisasi.

  Langkah terakhir dalam tahap ini adalah standarisasi fitur numerik yang terpilih, yaitu Glucose, BMI, dan Age. Proses standarisasi dilakukan menggunakan StandardScaler, yang bekerja dengan cara mengurangi nilai rata-rata (mean) dari setiap fitur dan membaginya dengan standar deviasi. Teknik ini diperlukan agar semua fitur memiliki skala distribusi yang seragam, sehingga model machine learning tidak bias terhadap fitur dengan skala nilai yang lebih besar.

## Modeling
Pada tahap ini, dilakukan proses pengembangan dan evaluasi model machine learning untuk menyelesaikan permasalahan prediksi Outcome pada dataset anemia. Tiga algoritma pembelajaran yang digunakan adalah K-Nearest Neighbors (KNN), Random Forest Regressor, dan AdaBoost Regressor (Boosting). Setiap model dibangun dan diuji menggunakan data hasil split sebelumnya, dan performanya dibandingkan menggunakan metrik Mean Squared Error (MSE) pada data pelatihan.

### 1. KNN 
Model KNN yang digunakan adalah KNeighborsRegressor dengan parameter n_neighbors=10. Model ini bekerja dengan menghitung jarak Euclidean antara titik data baru dan tetangga terdekat dari data pelatihan, lalu menghasilkan prediksi berdasarkan rata-rata nilai target dari tetangga tersebut.Parameter penting yang digunakan adalah n_neighbors=10 yang artinya prediksi akan didasarkan pada 10 tetangga terdekat. 

Kelebihan:
- Sederhana dan mudah diimplementasikan.
- Dapat digunakan untuk klasifikasi dan regresi.

Kekurangan:
- Sensitif terhadap skala dan outlier.
- Kurang efisien untuk dataset besar karena perlu menghitung jarak ke seluruh data latih setiap kali prediksi.

### 2. Random Forest 
Model kedua yang digunakan adalah RandomForestRegressor, yaitu metode ensemble yang membangun banyak pohon keputusan dan menggabungkan hasil prediksinya. Parameter yang digunakan beberapa diantaranya :
- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.


### 3. Boosting Algorithm
Model ketiga menggunakan AdaBoostRegressor, yaitu metode ensemble berbasis boosting yang melatih model secara bertahap, di mana setiap model baru difokuskan pada kesalahan prediksi model sebelumnya. Beberapa parameter yang digunakan : 
- learning_rate=0.05: mengatur kontribusi setiap model terhadap prediksi akhir.
- random_state=55: untuk memastikan reprodusibilitas hasil.


## Evaluation
Evaluasi dilakukan menggunakan metrik Mean Squared Error (MSE) pada data pelatihan. Hasil MSE masing-masing model disimpan dalam sebuah dataframe untuk perbandingan. Model dengan MSE terendah pada data pelatihan menunjukkan kemampuan terbaik dalam merepresentasikan data yang ada.

![image](https://github.com/user-attachments/assets/14d84ba3-61bf-40da-b8d1-d5a96400f56f)

MSE mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual. Semakin kecil, semakin baik performa model. KNN memiliki performa terbaik dengan MSE terkecil, menunjukkan hasil prediksi yang paling akurat dan perbedaan kecil antara nilai MSE di data train dan test menandakan model tidak overfitting. Random forest dan Boosting masih layak dipertimbangkan, namun akurasinya lebih rendah dari KNN.

![image](https://github.com/user-attachments/assets/61e78bcb-dace-4c55-8696-966162dc53d7)

Tabel di atas memperlihatkan perbandingan antara label sebenarnya (y_true) dan hasil prediksi dari tiga model klasifikasi yang digunakan: K-Nearest Neighbors (prediksi_KNN), Random Forest (prediksi_RF), dan Boosting (prediksi_Boosting). Pada baris yang ditampilkan, nilai aktual (y_true) adalah 1.
