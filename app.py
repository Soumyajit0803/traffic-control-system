import streamlit as st
import time
import random
import os
from datetime import datetime

st.set_page_config(page_title="Traffic Monitor", layout="centered")

st.markdown(
    "<h1 style='text-align: left; color: #fff; font-family: monospace;'>Traffic Light Simulation</h1>",
    unsafe_allow_html=True
)

IMAGE_FOLDER = "C:/Users/Brainstorm/Desktop/projects/traffic-controller/sample-traffic"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

placeholder = st.empty()

# Previous values
prev_green, prev_red = 30, 30

# Constants
S_MIN = 30
S_MAX = 120
ALPHA = 0.8     # traffic weight
BETA = 0.5      # pedestrian weight
GAMMA = 0.6     # irregularity weight
SMOOTHING = 0.8 # smoothing factor

def calculate_green_signal(traffic_density, people_density, irregularity_index, prev_green_time):
    """Adaptive green duration using traffic, pedestrian, and road irregularity."""
    # More potholes → slower road → longer green for safety
    raw_green = S_MIN + (S_MAX - S_MIN) * max(
        0, min(1, ALPHA * traffic_density - BETA * people_density + GAMMA * irregularity_index)
    )
    smooth_green = SMOOTHING * prev_green_time + (1 - SMOOTHING) * raw_green
    return round(smooth_green, 2)

def calculate_red_signal(traffic_density, people_density, irregularity_index, prev_red_time):
    """Adaptive red duration using traffic, pedestrian, and road irregularity."""
    # More irregularity → higher chance of slowdown → slightly longer red to manage flow
    raw_red = S_MIN + (S_MAX - S_MIN) * max(
        0, min(1, ALPHA * people_density - BETA * traffic_density + GAMMA * irregularity_index)
    )
    smooth_red = SMOOTHING * prev_red_time + (1 - SMOOTHING) * raw_red
    return round(smooth_red, 2)

while True:
    with placeholder.container():
        current_time = datetime.now().strftime("%H:%M:%S")
        people_density = random.random()
        traffic_density = random.uniform(0.7, 1.0)
        irregularity_index = random.uniform(0.2, 0.8) # Simulated road irregularity

        # 🚛 Truck restriction threshold
        TRUCK_IRREGULARITY_LIMIT = 0.6
        truck_allowed = irregularity_index < TRUCK_IRREGULARITY_LIMIT

        st.markdown(
            f"<p style='font-size: 1.2rem; text-align: left; color: #fff; font-family: monospace;'>Image captured at {current_time}</p>",
            unsafe_allow_html=True
        )

        signal_red = calculate_red_signal(traffic_density, people_density, irregularity_index, prev_red)
        signal_green = calculate_green_signal(traffic_density, people_density, irregularity_index, prev_green)

        prev_red, prev_green = signal_red, signal_green

        st.markdown(
            f"<p style='font-size: 1.2rem; text-align: left; color: #fff; font-family: monospace;'>Irregularity Index: {irregularity_index:.2f}</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-size: 1.2rem; text-align: left; color: #fff; font-family: monospace;'>Suggested Red Light Duration: {signal_red} s</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-size: 1.2rem; text-align: left; color: #fff; font-family: monospace;'>Suggested Green Light Duration: {signal_green} s</p>",
            unsafe_allow_html=True
        )

        if irregularity_index > 0.6:
            st.error("⚠️ Poor road condition detected — signal durations adjusted for safety!")
        elif signal_red > 45:
            st.error("🚦 Heavy traffic — longer red light needed!")
        elif signal_green < 30:
            st.success("🚶 More pedestrians — shorter green light recommended!")
        else:
            st.info("⚖️ Balanced flow and road conditions detected.")

        # 🚛 Truck restriction message
        if not truck_allowed:
            st.error("🚛 Heavy vehicles restricted! Road too uneven for safe passage.")
        else:
            st.success("✅ Trucks permitted — road within safe condition limits.")

    time.sleep(2)
    st.rerun()
