import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# =====================
# SETTING
# =====================

st.set_page_config(
    page_title="Analisis Statistik Aldi Frsh",
    layout="wide"
)


# =====================
# STYLE
# =====================

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#e3f2fd,#ffffff);
}

h1{
text-align:center;
}

</style>

""", unsafe_allow_html=True)



st.title("📊 Analisis Statistik Aldi Frsh")

st.write(
    "Upload data CSV atau Excel untuk analisis statistik"
)



# =====================
# UPLOAD
# =====================

file = st.file_uploader(
    "Upload File",
    type=["csv","xlsx"]
)



if file:


    if file.name.endswith(".csv"):

        df = pd.read_csv(file)

    else:

        df = pd.read_excel(file)



    st.subheader("Data Awal")

    st.dataframe(
        df,
        use_container_width=True
    )



    # =====================
    # INFORMASI
    # =====================


    st.header("Informasi Dataset")


    a,b,c = st.columns(3)


    a.metric(
        "Jumlah Data",
        len(df)
    )


    b.metric(
        "Jumlah Kolom",
        len(df.columns)
    )


    c.metric(
        "Missing Value",
        df.isnull().sum().sum()
    )



    # =====================
    # DESKRIPTIF
    # =====================


    st.header("Statistik Deskriptif")


    st.dataframe(
        df.describe()
    )



    # =====================
    # MISSING
    # =====================


    st.header("Missing Value")


    miss = pd.DataFrame({

        "Kolom":df.columns,

        "Jumlah Kosong":df.isnull().sum()

    })


    st.dataframe(miss)



    # =====================
    # VARIABEL ANGKA
    # =====================


    numerik = df.select_dtypes(
        include=np.number
    ).columns.tolist()



    if len(numerik) > 0:



        # GRAFIK

        st.header("Visualisasi")


        grafik = st.selectbox(
            "Pilih Variabel Grafik",
            numerik,
            key="grafik"
        )



        fig,ax = plt.subplots()


        sns.histplot(
            df[grafik],
            kde=True,
            ax=ax
        )


        st.pyplot(fig)



        # KORELASI


        st.header("Korelasi")


        if len(numerik)>1:


            fig,ax = plt.subplots(
                figsize=(10,6)
            )


            sns.heatmap(

                df[numerik].corr(),

                annot=True,

                ax=ax

            )


            st.pyplot(fig)


        else:

            st.warning(
                "Butuh minimal 2 variabel angka"
            )



        # NORMALITAS


        st.header("Uji Normalitas")


        normal = st.selectbox(

            "Pilih Variabel Normalitas",

            numerik,

            key="normal"

        )



        if st.button("Hitung Normalitas"):


            data = df[normal].dropna()



            if len(data)>500:

                data=data.sample(500)



            stat,p = shapiro(data)



            st.write(
                "P-value:",
                p
            )


            if p > 0.05:

                st.success(
                    "Data Normal"
                )

            else:

                st.warning(
                    "Data Tidak Normal"
                )



        # REGRESI


        st.header("Regresi Linear")


        x = st.selectbox(

            "Variabel X",

            numerik,

            key="x"

        )


        y = st.selectbox(

            "Variabel Y",

            numerik,

            key="y"

        )



        if st.button("Hitung Regresi"):


            model = LinearRegression()


            X = df[[x]]

            Y = df[y]


            model.fit(
                X,
                Y
            )


            pred = model.predict(X)


            r2 = r2_score(
                Y,
                pred
            )


            st.write(

                f"{y} = {model.coef_[0]:.4f}({x}) + {model.intercept_:.4f}"

            )


            st.write(
                "R² =",
                r2
            )



            fig,ax = plt.subplots()


            ax.scatter(
                X,
                Y
            )


            ax.plot(
                X,
                pred
            )


            st.pyplot(fig)



    else:


        st.warning(
            "Tidak ada data numerik"
        )



else:


    st.info(
        "Silahkan upload file"
    )
