import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import io

st.set_page_config(page_title="Oscillation Analyzer", layout="wide")
st.title("🔬 Oscillation Analyzer - Velocity Data Analysis Tool")

uploaded_file = st.file_uploader("Upload an Excel file with velocity data:", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    col_names = df.columns.tolist()

    if len(col_names) < 1:
        st.error("Excel file must have at least one column of velocity data.")
    else:
        st.success("File uploaded successfully!")

        # Extract velocity and define time vector
        velocity = df[col_names[0]].dropna().to_numpy()
        time = np.arange(len(velocity))  # Assuming 1 sample per time unit

        # Smooth velocity to reduce noise
        smoothed_velocity = uniform_filter1d(velocity, size=5)

        # Compute acceleration (numerical derivative)
        acceleration = np.gradient(smoothed_velocity)

        # Detect zero crossings for oscillation cycles
        zero_crossings = np.where(np.diff(np.sign(smoothed_velocity)))[0]
        num_oscillations = len(zero_crossings) // 2

        # Calculate range of motion and duration between zero-crossings
        ranges = []
        durations = []
        for i in range(0, len(zero_crossings) - 1, 2):
            segment = velocity[zero_crossings[i]:zero_crossings[i + 1]]
            if len(segment) > 0:
                range_of_motion = np.max(segment) - np.min(segment)
                duration = zero_crossings[i + 1] - zero_crossings[i]
                ranges.append(range_of_motion)
                durations.append(duration)

        # Calculate oscillation rates (cycles per sample)
        rates = [1 / d if d != 0 else 0 for d in durations]

        # Detect significant changes
        range_changes = np.where(np.abs(np.diff(ranges)) > np.std(ranges))[0]
        rate_changes = np.where(np.abs(np.diff(rates)) > np.std(rates))[0]

        # === Plotting ===
        st.subheader("📈 Plots")
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(time, velocity, label='Velocity')
        axs[0].set_title('Velocity')
        axs[0].set_ylabel('Velocity')
        axs[0].grid(True)

        axs[1].plot(time, acceleration, label='Acceleration', color='orange')
        axs[1].set_title('Acceleration')
        axs[1].set_ylabel('Acceleration')
        axs[1].set_xlabel('Time')
        axs[1].grid(True)

        st.pyplot(fig)

        # === Summary ===
        st.subheader("📊 Analysis Summary")
        st.write(f"**Total Oscillations Detected:** {num_oscillations}")
        st.write(f"**Average Range of Motion:** {np.mean(ranges):.2f}")
        st.write(f"**Average Oscillation Rate:** {np.mean(rates):.4f} cycles/sample")
        st.write(f"**Notable Range Changes at Cycles:** {range_changes.tolist()}")
        st.write(f"**Notable Rate Changes at Cycles:** {rate_changes.tolist()}")

        # === Prepare data for export ===
        df_velocity = pd.DataFrame({"Velocity": velocity})
        df_acceleration = pd.DataFrame({"Acceleration": acceleration})
        df_summary = pd.DataFrame({
            "Cycle": list(range(len(ranges))),
            "Range of Motion": ranges,
            "Oscillation Rate": rates
        })

        # === Export to Excel ===
        st.subheader("📤 Export Analysis")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_velocity.to_excel(writer, sheet_name='Velocity', index=False)
            df_acceleration.to_excel(writer, sheet_name='Acceleration', index=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Excel File",
            data=buffer,
            file_name="oscillation_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )