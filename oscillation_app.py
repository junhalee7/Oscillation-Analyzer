import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import openpyxl

# Define simps replacement using np.trapz
def simps(y, x):
    return np.trapz(y, x)

# Detect zero crossings
def detect_zero_crossings(velocity):
    return np.where(np.diff(np.sign(velocity)))[0]

# Compute acceleration
def compute_acceleration(velocity, time_step):
    return np.gradient(velocity, time_step)

# Analyze oscillations
def analyze_oscillations(time, velocity):
    crossings = detect_zero_crossings(velocity)
    motion_ranges = []
    durations = []
    rates = []

    for i in range(len(crossings) - 1):
        idx_start = crossings[i]
        idx_end = crossings[i + 1]
        v_segment = velocity[idx_start:idx_end + 1]
        t_segment = time[idx_start:idx_end + 1]

        if len(t_segment) < 2:
            continue

        range_of_motion = simps(np.abs(v_segment), t_segment)
        duration = t_segment[-1] - t_segment[0]
        rate = 1 / duration if duration > 0 else 0

        motion_ranges.append(range_of_motion)
        durations.append(duration)
        rates.append(rate)

    return motion_ranges, durations, rates

# Detect notable changes
def detect_notable_changes(values, threshold=0.3):
    values = np.array(values)
    avg = np.mean(values)
    return [i for i, val in enumerate(values) if abs(val - avg) > threshold * avg]

# Streamlit UI
st.title("Oscillation Analyzer")

uploaded_file = st.file_uploader("Upload Excel file with velocity data", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if 'velocity' not in df.columns:
        st.error("Excel file must have a column named 'velocity'.")
    else:
        velocity = df['velocity'].values
        time_step = 0.01  # assume fixed sampling interval (100Hz)
        time = np.arange(len(velocity)) * time_step
        acceleration = compute_acceleration(velocity, time_step)

        # Analyze oscillations
        motion_ranges, durations, rates = analyze_oscillations(time, velocity)

        # Detect notable changes
        range_changes = detect_notable_changes(motion_ranges)
        rate_changes = detect_notable_changes(rates)

        # Display plots
        st.subheader("Velocity")
        st.line_chart(velocity)

        st.subheader("Acceleration")
        st.line_chart(acceleration)

        st.subheader("Oscillation Summary")
        st.write(f"Total Oscillations Detected: {len(motion_ranges)}")

        df_out = pd.DataFrame({
            'Range of Motion': motion_ranges,
            'Duration (s)': durations,
            'Rate (Hz)': rates,
            'Notable Range Change': [i in range_changes for i in range(len(motion_ranges))],
            'Notable Rate Change': [i in rate_changes for i in range(len(rates))]
        })

        st.dataframe(df_out)

        # Export to Excel
        if st.button("Export Analysis to Excel"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                df_out.to_excel(tmp.name, index=False)
                tmp.flush()
                os.system(f"open '{tmp.name}'")  # macOS-specific: opens Excel file