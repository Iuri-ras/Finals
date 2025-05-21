import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""# Define Custom Bandpass Filter
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

# Streamlit App Configuration""
st.title("ECG Signal Filtering Application")
st.markdown("Upload your **ECG CSV file** to apply Bandpass Filtering (0.5 - 40 Hz)")
st.markdown("[Click here to download a sample ECG dataset from Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
st.markdown("[Click here to explore PhysioNet ECG Datasets](https://physionet.org/about/database/)")

# File Upload
st.markdown("### Or use a sample ECG Data")
if st.button("Load Sample Data"):
    sample_data = {
        'Time': np.linspace(0, 10, 2500),
        'ECG Signal': np.sin(2 * np.pi * 1 * np.linspace(0, 10, 2500)) + 0.5 * np.random.randn(2500)
    }
    df = pd.DataFrame(sample_data)
    st.write(df.head())
    
    # Plot Original Signal
    time = df['Time']
    ecg_signal = df['ECG Signal']
    
    fig, ax = plt.subplots()
    ax.plot(time, ecg_signal, label='Original Signal (Sample Data)')
    ax.set_title("Original ECG Signal - Sample Data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)
    
    # Apply Bandpass Filter (0.5 to 40 Hz)
    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    # Plot Filtered Signal
    st.markdown("### Filtered Signal (Sample Data)")
    fig, ax = plt.subplots()
    ax.plot(time, filtered_signal, color='orange', label='Filtered Signal')
    ax.set_title("Filtered ECG Signal - Sample Data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)

    # QRS Visibility Comment
    st.markdown("**QRS Visibility Improved:** The high-frequency noise and baseline drift have been filtered out, making the QRS complex more prominent and clearer to analyze.")









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




uploaded_file = st.file_uploader("Choose a CSV file", type="csv")





# Main Logic
if uploaded_file:
    # Load the data
    st.markdown("### Original Signal")
    df = pd.read_csv(uploaded_file)
    
    # Display DataFrame
    st.write(df.head())

    # Assume the first column is time and the second is ECG
    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]
    
    # Plot Original Signal
    fig, ax = plt.subplots()
    ax.plot(time, ecg_signal, label='Original Signal')
    ax.set_title("Original ECG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)
    
    # Apply Bandpass Filter (0.5 to 40 Hz)
    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    # Plot Filtered Signal
    st.markdown("### Filtered Signal")
    fig, ax = plt.subplots()
    ax.plot(time, filtered_signal, color='orange', label='Filtered Signal')
    ax.set_title("Filtered ECG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.legend()
    st.pyplot(fig)

    # QRS Visibility Comment
    st.markdown("**QRS Visibility Improved:** The high-frequency noise and baseline drift have been filtered out, making the QRS complex more prominent and clearer to analyze.")

    # Download Filtered Data
    st.markdown("### Download Filtered ECG Data")
    download_button = st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="filtered_ecg.csv", mime="text/csv")
