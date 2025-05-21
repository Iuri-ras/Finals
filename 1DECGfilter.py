# Display and process data if available
if df is not None:
    with st.expander("Preview Data"):
        st.write(df.head())

    time = df.iloc[:, 0]
    ecg_signal = df.iloc[:, 1]

    filtered_signal = custom_bandpass_filter(ecg_signal, 0.5, 40, fs=250)

    # Create two columns
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

    # Highlight QRS visibility improvement
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

    # Download filtered data
    filtered_df = pd.DataFrame({"Time": time, "Filtered ECG Signal": filtered_signal})
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered ECG Data as CSV",
        data=csv_data,
        file_name="filtered_ecg.csv",
        mime="text/csv",
        help="Download the filtered ECG signal data",
    )
