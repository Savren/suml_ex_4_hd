
import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "hd_model.sv"
model = pickle.load(open(filename, 'rb'))


# otwieramy wcześniej wytrenowany model

def main():
    st.set_page_config(page_title="Zawałotronic")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://scitechdaily.com/images/Man-Heart-Attack-Chest-Pain-Illustration.jpg")

    with overview:
        st.title("Zawałotronic")

    with left:
        symptom_slider = st.slider("Objawy", value=0, min_value=0, max_value=10, step=1)
        age_slider = st.slider("Wiek", value=10, min_value=10, max_value=100, step=1)
        sickness_slider = st.slider("Choroby współistniejące", value=0, min_value=0, max_value=10, step=1)

    with right:
        height_slider = st.slider("Wzrost", value=100, min_value=100, max_value=200)
        med_slider = st.slider("Leki", min_value=0, max_value=10, step=1)

    data = [[symptom_slider, age_slider, sickness_slider, height_slider, med_slider]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy takiej osobie grozi zawał?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
