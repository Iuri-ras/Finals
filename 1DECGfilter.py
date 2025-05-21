import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Custom Bandpass Filter (same as before)
def custom_bandpass_filter(data, lowcut, highcut, fs):
    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1/fs)
    mask = (frequencies > lowcut) & (frequencies < highcut)
    filtered_fft_data = np.zeros_like(fft_data)
    filtered_fft_data[mask] = fft_data[mask]
    filtered_signal = np.fft.ifft(filtered_fft_data).real
    return filtered_signal

st.title("🫀 ECG Signal Filtering Application")

# Group Upload and Load Sample in one box
with st.container():
    st.markdown("### Input Data")
    uploaded_file = st.file_uploader("Upload your ECG CSV file", type="csv")
    load_sample = st.button("Load Sample Data")

# Logic for loading sample or file
if load_sample:
    time = np.linspace(0, 10, 2500)
    ecg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.random.randn(2500)
    df = pd.DataFrame({"Time": time, "ECG Signal": ecg_signal})
    st.success("Sample ECG data loaded!")

elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name}")

else:
    df = None
    st.info("Upload a CSV file or load sample data to start.")

# The rest of your app follows here...

if df is not None:
    # Preview
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    st.subheader("Original ECG Signal")
    fig1, ax1 = plt.subplots()
    ax1.plot(time, ecg_signal, color="blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

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
            QRS Visibility Improved: Filtering removes baseline drift and noise, enhancing the QRS complex.
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
