import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================
# SETTING
# ==========================

st.set_page_config(
    page_title="Analisis Statistik Aldi Frsh",
    layout="wide"
)


# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#e3f2fd,#ffffff);
}

h1{
text-align:center;
font-size:40px;
}

</style>
""", unsafe_allow_html=True)



st.title("📊 Analisis Statistik Aldi Frsh")

st.write(
    "Aplikasi analisis data CSV dan Excel"
)



# ==========================
# UPLOAD DATA
# ==========================

file = st.file_uploader(
    "Upload File CSV / Excel",
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


    a,b,c = st.columns(3)


    a.metric(
        "Jumlah Data",
        len(df)
    )


    b.metric(
        "Jumlah Variabel",
        len(df.columns)
    )


    c.metric(
        "Missing Value",
        df.isnull().sum().sum()
    )



    # ==========================
    # DESKRIPTIF
    # ==========================


    st.header("📊 Statistik Deskriptif")


    des = df.describe()


    st.dataframe(des)



    # NARASI DESKRIPTIF


    kolom_awal = des.columns[0]


    st.info(
        f"""
        Dataset memiliki {len(df)} data.

        Variabel **{kolom_awal}** memiliki nilai rata-rata
        sebesar **{des.loc['mean',kolom_awal]:.2f}**.

        Nilai minimum sebesar 
        **{des.loc['min',kolom_awal]:.2f}**
        dan maksimum sebesar
        **{des.loc['max',kolom_awal]:.2f}**.
        """
    )



    # ==========================
    # MISSING VALUE
    # ==========================


    st.header("🔎 Missing Value")


    missing = pd.DataFrame({

        "Kolom":df.columns,

        "Jumlah Kosong":df.isnull().sum()

    })


    st.dataframe(missing)



    # ==========================
    # VARIABEL NUMERIK
    # ==========================


    numerik = df.select_dtypes(
        include=np.number
    ).columns.tolist()



    if len(numerik)>0:



        # ==========================
        # GRAFIK
        # ==========================


        st.header("📈 Visualisasi")


        pilih = st.selectbox(

            "Pilih Variabel",

            numerik,

            key="grafik"

        )


        fig,ax = plt.subplots()


        sns.histplot(

            df[pilih].dropna(),

            kde=True,

            ax=ax

        )


        st.pyplot(fig)



        # ==========================
        # KORELASI
        # ==========================


        st.header("🔥 Korelasi")


        if len(numerik)>1:


            fig,ax = plt.subplots(
                figsize=(10,6)
            )


            corr = df[numerik].corr()



            sns.heatmap(

                corr,

                annot=True,

                ax=ax

            )


            st.pyplot(fig)



            nilai = corr.abs().unstack()


            nilai = nilai[nilai < 1]


            pasangan = nilai.idxmax()


            hasil_corr = corr.loc[
                pasangan[0],
                pasangan[1]
            ]



            st.info(

                f"""
                Variabel **{pasangan[0]}**
                dan **{pasangan[1]}**

                memiliki hubungan korelasi sebesar
                **{hasil_corr:.2f}**.

                """

            )


        else:


            st.warning(
                "Data kurang untuk korelasi"
            )



        # ==========================
        # NORMALITAS
        # ==========================


        st.header("📋 Uji Normalitas")


        normal_var = st.selectbox(

            "Variabel Normalitas",

            numerik,

            key="normal"

        )



        if st.button("Hitung Normalitas"):


            data = df[normal_var].dropna()


            if len(data)>500:

                data=data.sample(500)



            stat,p = shapiro(data)



            st.write(
                "P-value:",
                p
            )


            if p>0.05:


                st.success(
                    "Data berdistribusi normal"
                )


            else:


                st.warning(
                    "Data tidak normal"
                )



            st.info(

                f"""
                Berdasarkan uji Shapiro-Wilk,
                nilai p-value adalah **{p:.4f}**.

                """

            )



        # ==========================
        # REGRESI
        # ==========================


        st.header("📈 Regresi Linear")


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


            data_reg = df[[x,y]].dropna()



            X = data_reg[[x]]

            Y = data_reg[y]



            model = LinearRegression()



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

                f"""
                Persamaan:

                **{y} = {model.coef_[0]:.4f}({x})
                + {model.intercept_:.4f}**

                """

            )


            st.write(
                "R² =",
                r2
            )



            st.info(

                f"""
                Variabel **{x}**
                mempengaruhi variabel **{y}**.

                Model mampu menjelaskan
                **{r2*100:.2f}%**
                variasi data.

                """

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
            "Tidak ada variabel angka"
        )



else:


    st.info(
        "Silahkan upload file"
    )
