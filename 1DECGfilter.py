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

# Initialize session state to track if sample data is loaded
if "load_sample_clicked" not in st.session_state:
    st.session_state.load_sample_clicked = False

# Custom CSS for sidebar styling and button pop-up effect
st.markdown("""
    <style>
    /* Sidebar container styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 2px 4px 10px rgb(0 0 0 / 0.1);
    }
    /* Section titles */
    .sidebar .sidebar-content h2, 
    .sidebar .sidebar-content h3 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 15px;
    }
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 10px !important;
        border: 1.5px solid #ced4da !important;
        padding: 10px !important;
        background-color: white !important;
        box-shadow: 0 0 4px rgba(0,0,0,0.05);
    }
    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
        outline: none;
    }
    /* Button pop-up on hover */
    .stButton > button:hover:not(:disabled) {
        transform: scale(1.05);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        cursor: pointer;
        background-color: #45a049;
    }
    /* Disabled button style */
    .stButton > button:disabled {
        background-color: #a5d6a7;
        cursor: not-allowed;
        transform: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("## 🫀 ECG Signal Filtering App")
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
        ax.plot(time, ecg_signal, color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Filtered ECG Signal")
        fig2, ax2 = plt.subplots()
        ax2.plot(time, filtered_signal, color="green")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        st.pyplot(fig2)

    st.markdown("""
        <div style="
            background-color:#198754;
            color: white;
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
