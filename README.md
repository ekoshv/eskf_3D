# IMU ESKF ZUPT Simulation Dashboard

A beautiful Streamlit web application for running and optimizing IMU Extended Kalman Filter (ESKF) with Zero Velocity Update (ZUPT) simulations.

## Features

- 🎯 **Interactive Simulation**: Run IMU ESKF simulations with customizable parameters
- ⚡ **Optimization**: Automatically find optimal filter gains using Optuna
- 📊 **Beautiful Visualizations**: Interactive 3D trajectories and 2D plots using Plotly
- 🎛️ **Parameter Control**: Easy-to-use sliders for all simulation parameters
- 📈 **Real-time Results**: Live updates of RMSE and parameter values

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Use the sidebar to adjust simulation parameters:
   - **General**: Total time, dt, random seed
   - **Noise**: Gyro, accelerometer, and GPS noise levels
   - **GPS**: GPS update rate and delay
   - **Trajectory Dynamics**: Frequency parameters for chaotic motion
   - **Attitude Dynamics**: Roll, pitch, yaw amplitudes and frequencies
   - **Optimization**: Number of optimization iterations

4. Click **"Run Simulation"** to execute the simulation with current parameters

5. Click **"Optimize"** to find the best filter gains (this may take several minutes)

6. View the interactive results including:
   - 3D trajectory visualization
   - Position, velocity, and attitude tracking plots
   - Real-time RMSE metrics

## How It Works

The application uses the `IMU_ESKF_ZUPT_V01.py` simulation without modification, simply importing it as a module. The web interface provides:

- **Parameter Configuration**: All simulation parameters are controllable via sliders
- **Simulation Execution**: Runs the ESKF with current parameters
- **Optimization**: Uses Optuna to find optimal filter gains by minimizing position RMSE
- **Visualization**: Creates interactive plots showing true vs estimated trajectories

## Tips

1. Start with default parameters to get familiar with the interface
2. Run optimization to find better filter gains for your specific scenario
3. After optimization, run the simulation again to see improved performance
4. Adjust trajectory dynamics parameters to create different motion profiles
5. Use higher GPS rates and lower noise for better tracking performance

Built with ❤️ using Streamlit and the IMU ESKF ZUPT simulation engine. 