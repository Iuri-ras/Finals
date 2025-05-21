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

# Initialize session state for Load Sample button
if "load_sample_clicked" not in st.session_state:
    st.session_state.load_sample_clicked = False

# Apply some custom CSS for styling
st.markdown(
    """
    <style>
    /* Style the sidebar upload/load section */
    .upload-section {
        background: #f5f7fa;
        padding: 25px 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .upload-section h2 {
        margin-top: 0;
        color: #4a4a4a;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 15px;
    }
    .stFileUploader>div {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 10px 15px;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover:not(:disabled) {
        background-color: #45a049;
        cursor: pointer;
    }
    .stButton>button:disabled {
        background-color: #a5d6a7;
        cursor: not-allowed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar title
st.sidebar.title("🫀 ECG Signal Filtering Application")

# Sidebar upload & load sample section inside a styled div
with st.sidebar:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("## Input Data")
    uploaded_file = st.file_uploader("Upload ECG CSV file", type="csv")
    load_sample = st.button(
        "Load Sample Data",
        disabled=st.session_state.load_sample_clicked,
        help="Load a generated sample ECG signal"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("Datasets")
    st.markdown("[Kaggle ECG Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
    st.markdown("[PhysioNet ECG Database](https://physionet.org/about/database/)")

# Page title and explanation
st.title("ECG Signal Filtering Application")
st.markdown("""
**What does the filter do?**

- Removes low-frequency baseline drift (<0.5 Hz) and high-frequency noise (>40 Hz).
- Enhances the clarity of the QRS complex, which is key for heart rate analysis.
""")

# Load sample data logic with disable after click
if load_sample:
    st.session_state.load_sample_clicked = True
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
        df = None
else:
    df = None
    st.info("Upload an ECG CSV file or load sample data to begin.")

# Data preview and plotting
if df is not None:
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    st.subheader("Original ECG Signal")
    fig, ax = plt.subplots()
    ax.plot(time, ecg_signal, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    st.subheader("Filtered ECG Signal")
    fig2, ax2 = plt.subplots()
    ax2.plot(time, filtered_signal, color="green")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    st.pyplot(fig2)

    st.markdown(
        """
        <div style="
            background-color:#198754;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            margin: 20px 0;
        ">
            QRS Visibility Improved: Filtering removes noise and drift, enhancing the QRS complex for analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered_df = pd.DataFrame({"Time": time, "Filtered ECG Signal": filtered_signal})
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered ECG Data as CSV",
        data=csv_data,
        file_name="filtered_ecg.csv",
        mime="text/csv",
        help="Download the filtered ECG signal data",
    )
