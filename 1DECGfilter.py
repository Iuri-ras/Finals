import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Bandpass filter function
def custom_bandpass_filter(data, lowcut, highcut, fs):
    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1/fs)
    mask = (frequencies > lowcut) & (frequencies < highcut)
    filtered_fft_data = np.zeros_like(fft_data)
    filtered_fft_data[mask] = fft_data[mask]
    filtered_signal = np.fft.ifft(filtered_fft_data).real
    return filtered_signal

# Apply custom CSS for your color palette
st.markdown(f"""
    <style>
    /* Sidebar background and text */
    .css-1d391kg, .css-1v3fvcr {{
        background-color: #344e41 !important;
        color: #dad7cd !important;
    }}
    /* Sidebar headers and labels */
    .css-1v3fvcr h2, .css-1v3fvcr h3, .css-1v3fvcr label, .css-1v3fvcr span {{
        color: #dad7cd !important;
    }}
    /* File uploader styling */
    .stFileUploader > div {{
        border-radius: 10px !important;
        border: 1.5px solid #588157 !important;
        padding: 10px !important;
        background-color: #3a5a40 !important;
        box-shadow: 0 0 8px rgba(163, 177, 138, 0.4);
        color: #dad7cd;
    }}
    /* Button styling */
    .stButton > button {{
        background-color: #588157 !important;
        color: #dad7cd !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
        outline: none;
        border: 1.5px solid #a3b18a;
    }}
    /* Button pop-up on hover */
    .stButton > button:hover:not(:disabled) {{
        transform: scale(1.05);
        box-shadow: 0 0 12px 3px rgba(163, 177, 138, 0.7);
        cursor: pointer;
        background-color: #a3b18a !important;
        color: #344e41 !important;
    }}
    /* Disabled button style */
    .stButton > button:disabled {{
        background-color: #dad7cd !important;
        cursor: not-allowed;
        transform: none !important;
        box-shadow: none !important;
        color: #588157 !important;
        border-color: #a3b18a !important;
    }}
    /* Text input and labels */
    input, label, span {{
        color: #dad7cd !important;
    }}
    /* Main page background & text */
    .css-18e3th9 {{
        background-color: #dad7cd !important;
        color: #344e41 !important;
    }}
    /* Headers on main page */
    h1, h2, h3, h4, h5, h6 {{
        color: #344e41 !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("## 🫀 ECG Signal Filtering")
st.sidebar.markdown("---")

with st.sidebar.container():
    st.markdown("### Input Options")
    uploaded_file = st.file_uploader("Upload your ECG CSV file", type="csv")
    load_sample = st.button("Load Sample Data")

st.sidebar.markdown("---")

st.sidebar.markdown("### Useful Datasets")
st.sidebar.markdown("[Kaggle ECG Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
st.sidebar.markdown("[PhysioNet ECG Database](https://physionet.org/about/database/)")

# Main app title and explanation
st.title("ECG Signal Filtering Application")
st.markdown("""
**What does the filter do?**

- Removes low-frequency baseline drift (<0.5 Hz) and high-frequency noise (>40 Hz).
- Enhances the clarity of the QRS complex, which is key for heart rate analysis.
""")

# Initialize df variable early
df = None

# Load sample or uploaded file
if load_sample:
    time = np.linspace(0, 10, 2500)
    ecg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.random.randn(2500)
    df = pd.DataFrame({"Time": time, "ECG Signal": ecg_signal})
    st.success("Sample ECG data loaded!")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload an ECG CSV file or load sample data to begin.")

# Display and process data if available
if df is not None:
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original ECG Signal")
        fig, ax = plt.subplots()
        ax.plot(time, ecg_signal, color="#344e41")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Filtered ECG Signal")
        fig2, ax2 = plt.subplots()
        ax2.plot(time, filtered_signal, color="#588157")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        st.pyplot(fig2)

    st.markdown(f"""
        <div style="
            background-color:{'#588157'};
            color: { '#dad7cd' };
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            margin: 20px 0;">
            QRS Visibility Improved: Filtering removes noise and drift, enhancing the QRS complex for analysis.
        </div>
        """, unsafe_allow_html=True)

    filtered_df = pd.DataFrame({"Time": time, "Filtered ECG Signal": filtered_signal})
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered ECG Data as CSV",
        data=csv_data,
        file_name="filtered_ecg.csv",
        mime="text/csv",
        help="Download the filtered ECG signal data",
    )
