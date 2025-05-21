import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Custom Bandpass Filter
def custom_bandpass_filter(data, lowcut, highcut, fs):
    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1/fs)

    # Create a mask for frequencies within the bandpass range
    mask = (frequencies > lowcut) & (frequencies < highcut)
    filtered_fft_data = np.zeros_like(fft_data)
    filtered_fft_data[mask] = fft_data[mask]

    # Inverse FFT to get the filtered time-domain signal
    filtered_signal = np.fft.ifft(filtered_fft_data).real
    return filtered_signal

# Streamlit App Configuration
st.title("🫀 ECG Signal Filtering Application")
st.markdown("Upload your **ECG CSV file** to apply Bandpass Filtering (0.5 - 40 Hz).")
st.markdown("[Click here to download a sample ECG dataset from Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
st.markdown("[Click here to explore PhysioNet ECG Datasets](https://physionet.org/about/database/)")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file with ECG data", type="csv")

# Add Load Sample Data Button
load_sample = st.button("Load Sample Data")

if load_sample:
    # Sample Data Creation
    sample_data = {
        'Time': np.linspace(0, 10, 2500),
        'ECG Signal': np.sin(2 * np.pi * 1 * np.linspace(0, 10, 2500)) + 0.5 * np.random.randn(2500)
    }
    df = pd.DataFrame(sample_data)
    st.success("Sample ECG data loaded!")

elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name}")

else:
    df = None
    st.info("Please upload an ECG CSV file or click 'Load Sample Data'.")

# Proceed if data is loaded
if df is not None:
    # Show first rows in an expandable section
    with st.expander("Preview Data"):
        st.write(df.head())

    # Assume first col = time, second col = ECG signal
    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    # Plot Original ECG Signal
    st.subheader("Original ECG Signal")
    fig1, ax1 = plt.subplots()
    ax1.plot(time, ecg_signal, label="Original Signal", color='blue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    st.pyplot(fig1)

    # Apply Bandpass Filter (0.5 to 40 Hz)
    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    # Plot Filtered Signal
    st.subheader("Filtered ECG Signal")
    fig2, ax2 = plt.subplots()
    ax2.plot(time, filtered_signal, label="Filtered Signal", color='green')  # Filtered signal is green now
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    st.pyplot(fig2)

    # QRS Visibility Comment
    st.markdown(
        """
        <div style='
            background-color:#4CAF50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            text-align: center;
            margin-top: 15px;
            margin-bottom: 15px;
        '>
            QRS Visibility Improved: The filtering reduces noise and baseline drift, making the QRS complex clearer for analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Download filtered ECG data
    filtered_df = pd.DataFrame({"Time": time, "Filtered ECG Signal": filtered_signal})
    csv = filtered_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Filtered ECG Data as CSV",
        data=csv,
        file_name='filtered_ecg.csv',
        mime='text/csv',
        help="Download the filtered ECG signal data"
    )
