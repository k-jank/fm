import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import altair as alt
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = CatBoostClassifier()
model.load_model('catboost_model.bin')

# Widget untuk mengunggah file Excel
uploaded_file = st.sidebar.file_uploader("Upload Excel file to analyze players", type=["xlsx", "xls"])

if uploaded_file is None:
    xls = pd.ExcelFile('feature.xlsx')
    sheet_names_upper = [sheet.upper() for sheet in xls.sheet_names]
    sheet_names_upper.sort() 
    selected_sheet = st.sidebar.selectbox("Feature Importance for", [sheet.upper() for sheet in sheet_names_upper])
    selected_sheet_lower = xls.sheet_names[sheet_names_upper.index(selected_sheet)].lower()
    feature_importance_data = pd.read_excel('feature.xlsx', sheet_name=selected_sheet_lower)
    feature_importance_data_sorted = feature_importance_data.sort_values(by='feature_importance', ascending=False)

    # Set title for the chart
    st.markdown(f"<h2 style='text-align: center;'>Feature Importance for {selected_sheet}</h2>", unsafe_allow_html=True)
    chart = alt.Chart(feature_importance_data_sorted).mark_bar().encode(
        y=alt.Y('feature_name', sort='-x'),  # Sorting by feature importance descending
        x='feature_importance'
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

# Jika file telah diunggah
if uploaded_file is not None:
    try:
        # Membaca data dari file Excel
        data = pd.read_excel(uploaded_file)
        data = data[model.feature_names_]

        # Make predictions using the model
        predictions = model.predict(data)

        # Convert the nested list to a flat list
        predictions_flat = predictions.flatten()

        # Data
        fm = pd.read_excel(uploaded_file)
        hasil_prediksi_df = pd.DataFrame(fm)
        hasil_prediksi_df['Best Role'] = predictions_flat

        # Tampilkan dropdown untuk memilih posisi
        selected_position = st.sidebar.selectbox("Choose Player Role:", sorted(hasil_prediksi_df['Best Role'].unique()))

        # Filter DataFrame based on selected position
        filtered_df = hasil_prediksi_df[hasil_prediksi_df['Best Role'] == selected_position]
        unique_name = sorted(filtered_df['Name'].unique())
        selected_players = st.sidebar.multiselect("Players:", unique_name)

        # Handle comparison between selected players
        if selected_players:
            # Display player information for all selected players
            st.markdown(f"<h1 style='text-align: center;'>Football Manager Player Analyzer</h1>", unsafe_allow_html=True)
            num_players = len(selected_players)
            cols = st.columns(num_players)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Example colors
            for i, selected_player in enumerate(selected_players):
                selected_data = filtered_df[filtered_df['Name'] == selected_player]
                with cols[i]:
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: black; font-weight: bold;'><b>{selected_player}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>Best Role</b><br>{selected_data['Best Role'].iloc[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>Age</b><br>{selected_data['Age'].iloc[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>Height</b><br>{selected_data['Height'].iloc[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>Weight</b><br>{selected_data['Weight'].iloc[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>CA</b><br>{selected_data['CA'].iloc[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {colors[i % len(colors)]}; color: white;'><b>PA</b><br>{selected_data['PA'].iloc[0]}</div>", unsafe_allow_html=True)

            # Generate radar plot for all selected players
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            num_cols = len(filtered_df.columns) - 7 
            angles = np.linspace(0, 2 * np.pi, num_cols, endpoint=False).tolist()
            for selected_player in selected_players:
                selected_data = filtered_df[filtered_df['Name'] == selected_player]
                selected_columns = selected_data.drop(columns=['Name', 'Age', 'CA', 'PA', 'Height', 'Weight', 'Best Role', ])
                values = selected_columns.iloc[0].values.tolist()
                ax.fill(angles, values, alpha=0.25, label=selected_player)
            max_value = selected_columns.max().max()
            interval = max_value / 5 
            for i in range(5):
                ax.text(angles[i] - np.pi/num_cols, interval * (i + 1), str(int(round(interval * (i + 1)))), ha='center', va='center')
            ax.set_xticks(angles)
            ax.set_xticklabels(selected_columns.columns)
            ax.set_yticklabels([])
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='x-small')
            st.write("")
            ax.set_title('Player Attributes', pad=20)
            st.pyplot(fig)
            plt.close(fig) 


    except Exception as e:
        st.error(f"An error occurred: {e}")