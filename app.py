import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import optuna
import time
from IMU_ESKF_ZUPT_V01 import IMUSimulator, objective

# Configure page
st.set_page_config(
    page_title="IMU ESKF ZUPT Simulation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'optimized_params' not in st.session_state:
        st.session_state.optimized_params = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'last_rmse' not in st.session_state:
        st.session_state.last_rmse = None
    if 'optimization_study' not in st.session_state:
        st.session_state.optimization_study = None

initialize_session_state()

# Title and description
st.title("🚀 IMU ESKF ZUPT Simulation Dashboard")
st.markdown("---")

# Sidebar for parameters
with st.sidebar:
    st.header("📊 Simulation Parameters")
    
    # General parameters
    st.subheader("General")
    total_time = st.slider("Total Time (s)", min_value=10.0, max_value=200.0, value=60.0, step=1.0)
    dt = st.slider("dt (s)", min_value=0.005, max_value=0.1, value=0.01, step=0.005, format="%.3f")
    random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
    
    # Noise parameters
    st.subheader("Noise")
    gyro_noise_std = st.slider("Gyro Noise Std", min_value=0.0001, max_value=1.0, value=0.05, step=0.0001, format="%.4f")
    accel_noise_std = st.slider("Accel Noise Std", min_value=0.0001, max_value=1.0, value=0.002, step=0.0001, format="%.4f")
    gps_noise_std = st.slider("GPS Noise Std", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    
    # GPS parameters
    st.subheader("GPS")
    gps_rate = st.slider("GPS Rate (Hz)", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
    gps_delay = st.slider("GPS Delay (s)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    # Dynamics parameters
    st.subheader("Trajectory Dynamics")
    freq_x1 = st.slider("Freq X1", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
    freq_x2 = st.slider("Freq X2", min_value=0.1, max_value=5.0, value=0.6, step=0.1)
    freq_y1 = st.slider("Freq Y1", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    freq_y2 = st.slider("Freq Y2", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
    freq_z1 = st.slider("Freq Z1", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    freq_z2 = st.slider("Freq Z2", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    
    # Attitude parameters
    st.subheader("Attitude Dynamics")
    amp_roll = st.slider("Amp Roll", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    amp_pitch = st.slider("Amp Pitch", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    amp_yaw = st.slider("Amp Yaw", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    freq_roll = st.slider("Freq Roll", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    freq_pitch = st.slider("Freq Pitch", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    freq_yaw = st.slider("Freq Yaw", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    # Optimization parameters
    st.subheader("Optimization")
    n_trials = st.slider("Number of Optimization Iterations", min_value=10, max_value=200, value=50, step=10)

# Main content area
col1, col2 = st.columns([1, 1])

# Prepare parameters dictionary
params = {
    'dt': dt,
    'T_total': total_time,
    'gps_rate': gps_rate,
    'gps_delay': gps_delay,
    'gyro_noise_std': gyro_noise_std,
    'accel_noise_std': accel_noise_std,
    'gps_noise_std': gps_noise_std,
    'seed': random_seed,
    'freq_x1': freq_x1,
    'freq_x2': freq_x2,
    'freq_y1': freq_y1,
    'freq_y2': freq_y2,
    'freq_z1': freq_z1,
    'freq_z2': freq_z2,
    'amp_roll': amp_roll,
    'amp_pitch': amp_pitch,
    'amp_yaw': amp_yaw,
    'freq_roll': freq_roll,
    'freq_pitch': freq_pitch,
    'freq_yaw': freq_yaw
}

# Buttons
with col1:
    if st.button("🎯 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            simulator = IMUSimulator(**params)
            
            # Use optimized parameters if available, otherwise use default values
            if st.session_state.optimized_params:
                gains = st.session_state.optimized_params
                gps_alpha = gains['gps_alpha']
            else:
                # Provide default gains when no optimization has been done
                gains = {
                    'gyro_factor': 1.0,
                    'accel_factor': 1.0,
                    'gps_alpha': 0.5,
                    'est_alpha': 0.5
                }
                gps_alpha = 0.5
            
            rmse = simulator.run_simulation(gains=gains, plot_results=False, gps_alpha=gps_alpha)
            st.session_state.simulation_results = simulator
            st.session_state.last_rmse = rmse
        
        st.success(f"✅ Simulation completed! RMSE: {rmse:.3f} m")

with col2:
    optimize_placeholder = st.empty()
    
    if optimize_placeholder.button("⚡ Optimize", type="secondary"):
        with st.spinner(f"Optimizing with {n_trials} trials..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            study = optuna.create_study(direction='minimize')
            
            for i in range(n_trials):
                study.optimize(lambda trial: objective(trial, params), n_trials=1)
                progress = (i + 1) / n_trials
                progress_bar.progress(progress)
                status_text.text(f"Trial {i + 1}/{n_trials} - Best RMSE: {study.best_value:.3f}")
            
            st.session_state.optimized_params = study.best_params
            st.session_state.optimization_study = study
            progress_bar.empty()
            status_text.empty()
        
        st.success(f"✅ Optimization completed! Best RMSE: {study.best_value:.3f} m")

# Display optimized parameters if available
if st.session_state.optimized_params:
    st.subheader("🎯 Optimized Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gyro Factor", f"{st.session_state.optimized_params['gyro_factor']:.4f}")
    with col2:
        st.metric("Accel Factor", f"{st.session_state.optimized_params['accel_factor']:.4f}")
    with col3:
        st.metric("GPS Alpha", f"{st.session_state.optimized_params['gps_alpha']:.3f}")
    with col4:
        st.metric("Est Alpha", f"{st.session_state.optimized_params['est_alpha']:.3f}")

# Display current RMSE
if st.session_state.last_rmse:
    st.metric("Current Position RMSE", f"{st.session_state.last_rmse:.3f} m")

# Display simulation results
if st.session_state.simulation_results:
    simulator = st.session_state.simulation_results
    
    st.subheader("📈 Simulation Results")
    
    # Create plots
    time = simulator.time
    pos_true = simulator.x_true_hist[:, 7:10]
    pos_est = simulator.x_est_hist[:, 7:10]
    vel_true = simulator.x_true_hist[:, 4:7]
    vel_est = simulator.x_est_hist[:, 4:7]
    
    # Convert quaternions to Euler angles
    def euler_from_quaternion(q):
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return roll, pitch, yaw
    
    true_euler = np.array([euler_from_quaternion(simulator.x_true_hist[k, 0:4]) for k in range(len(time))])
    est_euler = np.array([euler_from_quaternion(simulator.x_est_hist[k, 0:4]) for k in range(len(time))])
    
    # 3D Trajectory Plot
    st.subheader("🌍 3D Trajectory")
    fig_3d = go.Figure()
    
    fig_3d.add_trace(go.Scatter3d(
        x=pos_true[:, 0], y=pos_true[:, 1], z=pos_true[:, 2],
        mode='lines', name='True Trajectory',
        line=dict(color='blue', width=4)
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=pos_est[:, 0], y=pos_est[:, 1], z=pos_est[:, 2],
        mode='lines', name='Estimated Trajectory',
        line=dict(color='red', width=4)
    ))
    
    # Add GPS measurements
    gps_idx = ~np.isnan(simulator.z_gps[:, 0])
    if np.any(gps_idx):
        gps_positions = simulator.z_gps[gps_idx]
        fig_3d.add_trace(go.Scatter3d(
            x=gps_positions[:, 0], y=gps_positions[:, 1], z=gps_positions[:, 2],
            mode='markers', name='GPS Measurements',
            marker=dict(size=4, color='green', opacity=0.5)
        ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube'
        ),
        title=f"3D Trajectory Comparison (RMSE: {st.session_state.last_rmse:.3f} m)",
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Position plots
    st.subheader("📍 Position Tracking")
    fig_pos = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X Position', 'Y Position', 'Z Position'),
        vertical_spacing=0.08
    )
    
    # X Position
    fig_pos.add_trace(go.Scatter(x=time, y=pos_true[:, 0], mode='lines', name='True X', line=dict(color='blue')), row=1, col=1)
    fig_pos.add_trace(go.Scatter(x=time, y=pos_est[:, 0], mode='lines', name='Est X', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Y Position
    fig_pos.add_trace(go.Scatter(x=time, y=pos_true[:, 1], mode='lines', name='True Y', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig_pos.add_trace(go.Scatter(x=time, y=pos_est[:, 1], mode='lines', name='Est Y', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
    
    # Z Position
    fig_pos.add_trace(go.Scatter(x=time, y=pos_true[:, 2], mode='lines', name='True Z', line=dict(color='blue'), showlegend=False), row=3, col=1)
    fig_pos.add_trace(go.Scatter(x=time, y=pos_est[:, 2], mode='lines', name='Est Z', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
    
    fig_pos.update_layout(height=800, title_text="Position Components vs Time")
    fig_pos.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig_pos.update_yaxes(title_text="X (m)", row=1, col=1)
    fig_pos.update_yaxes(title_text="Y (m)", row=2, col=1)
    fig_pos.update_yaxes(title_text="Z (m)", row=3, col=1)
    
    st.plotly_chart(fig_pos, use_container_width=True)
    
    # Velocity plots
    st.subheader("🏃 Velocity Tracking")
    fig_vel = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X Velocity', 'Y Velocity', 'Z Velocity'),
        vertical_spacing=0.08
    )
    
    # X Velocity
    fig_vel.add_trace(go.Scatter(x=time, y=vel_true[:, 0], mode='lines', name='True Vx', line=dict(color='blue')), row=1, col=1)
    fig_vel.add_trace(go.Scatter(x=time, y=vel_est[:, 0], mode='lines', name='Est Vx', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Y Velocity
    fig_vel.add_trace(go.Scatter(x=time, y=vel_true[:, 1], mode='lines', name='True Vy', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig_vel.add_trace(go.Scatter(x=time, y=vel_est[:, 1], mode='lines', name='Est Vy', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
    
    # Z Velocity
    fig_vel.add_trace(go.Scatter(x=time, y=vel_true[:, 2], mode='lines', name='True Vz', line=dict(color='blue'), showlegend=False), row=3, col=1)
    fig_vel.add_trace(go.Scatter(x=time, y=vel_est[:, 2], mode='lines', name='Est Vz', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
    
    fig_vel.update_layout(height=800, title_text="Velocity Components vs Time")
    fig_vel.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig_vel.update_yaxes(title_text="Vx (m/s)", row=1, col=1)
    fig_vel.update_yaxes(title_text="Vy (m/s)", row=2, col=1)
    fig_vel.update_yaxes(title_text="Vz (m/s)", row=3, col=1)
    
    st.plotly_chart(fig_vel, use_container_width=True)
    
    # Attitude plots
    st.subheader("🧭 Attitude Tracking")
    fig_att = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Roll', 'Pitch', 'Yaw'),
        vertical_spacing=0.08
    )
    
    # Roll
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(true_euler[:, 0]), mode='lines', name='True Roll', line=dict(color='blue')), row=1, col=1)
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(est_euler[:, 0]), mode='lines', name='Est Roll', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Pitch
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(true_euler[:, 1]), mode='lines', name='True Pitch', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(est_euler[:, 1]), mode='lines', name='Est Pitch', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)
    
    # Yaw
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(true_euler[:, 2]), mode='lines', name='True Yaw', line=dict(color='blue'), showlegend=False), row=3, col=1)
    fig_att.add_trace(go.Scatter(x=time, y=np.degrees(est_euler[:, 2]), mode='lines', name='Est Yaw', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)
    
    fig_att.update_layout(height=800, title_text="Attitude (Euler Angles) vs Time")
    fig_att.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig_att.update_yaxes(title_text="Roll (deg)", row=1, col=1)
    fig_att.update_yaxes(title_text="Pitch (deg)", row=2, col=1)
    fig_att.update_yaxes(title_text="Yaw (deg)", row=3, col=1)
    
    st.plotly_chart(fig_att, use_container_width=True)

# Instructions
st.markdown("---")
st.subheader("📖 Instructions")
st.markdown("""
1. **Adjust Parameters**: Use the sidebar sliders to configure simulation parameters
2. **Run Simulation**: Click "Run Simulation" to execute with current parameters
3. **Optimize**: Click "Optimize" to find the best filter gains (this may take a few minutes)
4. **View Results**: Interactive plots will show the comparison between true and estimated trajectories
5. **Iterate**: After optimization, run the simulation again to see improved results
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | IMU ESKF ZUPT Simulation Dashboard") 