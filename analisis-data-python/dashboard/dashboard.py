import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
day_df = pd.read_csv("data/day.csv")
day_df["dteday"] = pd.to_datetime(day_df["dteday"])
day_df.set_index("dteday", inplace=True)

st.title("Dashboard Analisis Bike Sharing")
st.subheader("Tabel Dataframe")

# Fitur interaktif berdasarkan tanggal
st.sidebar.subheader("Filter Data")
start = st.sidebar.date_input("Tanggal Mulai", day_df.index.min().date())
end = st.sidebar.date_input("Tanggal Akhir", day_df.index.max().date())

# Visualisasi berdasarkan tanggal 
if start > end:
    st.sidebar.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir!")
else:
    fiter_df = day_df.loc[start:end]

    st.write(f"Data dari {start} sampai {end}")
    if fiter_df.empty:
        st.warning("Tidak ada data dalam rentang tanggal yang dipilih!")
    else:
        st.write(fiter_df.head())

        month_corr = fiter_df.resample('M').agg({
            "temp": "mean",
            "atemp": "mean",
            "windspeed": "mean",
            "cnt": "sum"
        })

        if month_corr.empty:
            st.warning("Data setelah resampling kosong. Coba pilih rentang tanggal yang lebih luas.")
        else:
            st.write("Korelasi antara variabel:")
            st.write(month_corr.corr())

            # Visualisasi 1: Pengaruh Suhu terhadap Peminjaman Sepeda
            st.subheader("Visualisasi Data")
            st.write("Pengaruh Suhu terhadap Peminjaman Sepeda")
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(month_corr.index, 
                     month_corr["cnt"], 
                     marker="o",
                     color="b",
                     label="Total Peminjaman")
            ax1.set_xlabel("Bulan")
            ax1.set_ylabel("Total Peminjaman", color="b")

            ax2 = ax1.twinx()
            ax2.plot(month_corr.index, 
                    month_corr["temp"], 
                    marker="s",
                    color="r",
                    label="Rata-rata Suhu")
            ax2.set_ylabel("Rata-rata Suhu", color="r")
            plt.title("Pengaruh Suhu dan Peminjaman Sepeda per Bulan")
            st.pyplot(fig)

            # Visualisasi 2: Pengaruh Kecepatan Angin terhadap Peminjaman Sepeda**
            st.write("Pengaruh Kecepatan Angin terhadap Peminjaman Sepeda")
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(month_corr.index, 
                     month_corr["cnt"], 
                     marker="o",
                     color="b", 
                     label="Total Peminjaman")
            ax1.set_xlabel("Bulan")
            ax1.set_ylabel("Total Peminjaman", color="b")

            ax2 = ax1.twinx()
            ax2.plot(month_corr.index, 
                     month_corr["windspeed"], 
                     marker="s",
                     color="r", 
                     alpha=0.6, 
                     label="Rata-rata Kecepatan Angin")
            ax2.set_ylabel("Rata-rata Kecepatan Angin", color="r")
            plt.title("Pengaruh Kecepatan Angin dan Peminjaman Sepeda")
            st.pyplot(fig)
