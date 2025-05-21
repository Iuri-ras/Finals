import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Custom Bandpass Filter once at the top
def custom_bandpass_filter(data, lowcut, highcut, fs):
    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1/fs)

    mask = (frequencies > lowcut) & (frequencies < highcut)
    filtered_fft_data = np.zeros_like(fft_data)
    filtered_fft_data[mask] = fft_data[mask]

    filtered_signal = np.fft.ifft(filtered_fft_data).real
    return filtered_signal

# Sidebar for datasets and info
with st.sidebar:
    st.header("ECG Datasets")
    st.markdown("[Kaggle ECG Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
    st.markdown("[PhysioNet ECG Database](https://physionet.org/about/database/)")
    st.markdown("---")
    st.write("Developed with Streamlit")

# Main title and explanation
st.title("🫀 ECG Signal Filtering Application")

st.markdown("""
This application filters ECG signals to remove noise and baseline wander, enhancing the visibility of the **QRS complex**,  
which is crucial for heart rate and rhythm analysis.

**What is filtering doing?**  
- **Bandpass filtering** allows frequencies between 0.5 Hz and 40 Hz to pass through.  
- This removes low-frequency baseline drift and high-frequency noise (like powerline interference).  
- The result is a cleaner ECG signal, making important features like QRS complexes easier to detect and analyze.
""")

# Upload and Sample buttons side-by-side
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload ECG CSV file", type="csv")
with col2:
    load_sample = st.button("Load Sample Data")

# Load sample data if button clicked
if load_sample:
    time = np.linspace(0, 10, 2500)
    ecg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.random.randn(2500)
    df = pd.DataFrame({"Time": time, "ECG Signal": ecg_signal})
    st.success("Sample ECG data loaded!")

# If uploaded file provided, load it
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name}")

else:
    df = None
    st.info("Upload a CSV file or load sample data to get started.")

# If we have data, process and display
if df is not None:
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    st.subheader("Original ECG Signal")
    fig1, ax1 = plt.subplots()
    ax1.plot(time, ecg_signal, label="Original Signal", color="blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    st.pyplot(fig1)

    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    st.subheader("Filtered ECG Signal")
    fig2, ax2 = plt.subplots()
    ax2.plot(time, filtered_signal, label="Filtered Signal", color="green")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
        """
        <div style='
            background-color:#198754;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        '>
            QRS Visibility Improved: Filtering reduces noise and baseline drift, enhancing the QRS complex for clearer analysis.
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
        help="gago the filtered ECG signal data",
    )
