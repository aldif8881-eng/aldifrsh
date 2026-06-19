import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================
# SETTING HALAMAN
# ==========================

st.set_page_config(
    page_title="Analisis Statistik Aldi Frsh",
    layout="wide"
)


# ==========================
# STYLE MODERN
# ==========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#e3f2fd,#f5f5f5);
}

h1 {
    text-align:center;
    font-size:40px;
    font-weight:bold;
}

.block-container{
    padding-top:2rem;
}

</style>

""", unsafe_allow_html=True)



st.title("📊 Analisis Statistik Aldi Frsh")

st.write(
    "Aplikasi analisis data statistik menggunakan Streamlit"
)



# ==========================
# UPLOAD CSV / EXCEL
# ==========================

file = st.file_uploader(
    "📂 Upload File Data (CSV / Excel)",
    type=["csv","xlsx"]
)



if file:


    if file.name.endswith(".csv"):

        df = pd.read_csv(file)

    else:

        df = pd.read_excel(file)



    st.subheader("📄 Data Awal")

    st.dataframe(
        df,
        use_container_width=True
    )



    # ==========================
    # INFORMASI
    # ==========================


    st.header("📌 Informasi Dataset")


    col1,col2,col3 = st.columns(3)


    col1.metric(
        "Jumlah Data",
        len(df)
    )


    col2.metric(
        "Jumlah Variabel",
        len(df.columns)
    )


    col3.metric(
        "Missing Value",
        df.isnull().sum().sum()
    )



    # ==========================
    # DESKRIPTIF
    # ==========================


    st.header("📊 Statistik Deskriptif")


    st.dataframe(
        df.describe(),
        use_container_width=True
    )



    # ==========================
    # MISSING VALUE
    # ==========================


    st.header("🔎 Cek Missing Value")


    miss=pd.DataFrame(
        {
        "Kolom":df.columns,
        "Jumlah Kosong":df.isnull().sum()
        }
    )


    st.table(miss)



    # ==========================
    # VISUALISASI
    # ==========================


    st.header("📈 Visualisasi")


    numerik=df.select_dtypes(
        include=np.number
    ).columns



    pilih=st.selectbox(
        "Pilih Variabel",
        numerik
    )


    fig,ax=plt.subplots()


    sns.histplot(
        df[pilih],
        kde=True,
        ax=ax
    )


    st.pyplot(fig)



    # ==========================
    # KORELASI
    # ==========================


    st.header("🔥 Korelasi Variabel")


    fig,ax=plt.subplots(
        figsize=(10,6)
    )


    sns.heatmap(
        df[numerik].corr(),
        annot=True,
        ax=ax
    )


    st.pyplot(fig)



    # ==========================
    # NORMALITAS
    # ==========================


    st.header("📋 Uji Normalitas Shapiro")


    variabel=st.selectbox(
        "Variabel Normalitas",
        numerik
    )


    if st.button("Hitung Normalitas"):


        data=df[variabel].dropna()


        if len(data)>500:

            data=data.sample(500)



        stat,p=shapiro(data)



        st.write(
            "P-value:",
            p
        )


        if p>0.05:

            st.success(
                "Data Berdistribusi Normal"
            )

        else:

            st.warning(
                "Data Tidak Normal"
            )



    # ==========================
    # REGRESI
    # ==========================


    st.header("📈 Regresi Linear")


    x=st.selectbox(
        "Variabel Bebas (X)",
        numerik
    )


    y=st.selectbox(
        "Variabel Terikat (Y)",
        numerik
    )



    if st.button("Analisis Regresi"):


        model=LinearRegression()


        X=df[[x]]

        Y=df[y]


        model.fit(
            X,
            Y
        )


        pred=model.predict(X)



        r2=r2_score(
            Y,
            pred
        )


        st.write(
            f"Persamaan:"
        )


        st.write(
            f"{y} = {model.coef_[0]:.4f}({x}) + {model.intercept_:.4f}"
        )


        st.write(
            "R² = ",
            r2
        )



        fig,ax=plt.subplots()


        ax.scatter(
            X,
            Y
        )


        ax.plot(
            X,
            pred
        )


        ax.set_xlabel(x)

        ax.set_ylabel(y)


        st.pyplot(fig)



else:


    st.info(
        "Silahkan upload file CSV atau Excel"
    )