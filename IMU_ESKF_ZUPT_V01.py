import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import optuna

pio.renderers.default = 'browser'

#############################
# Helper Functions
#############################

def quaternion_mult(q, p):
    # q and p are 4-vectors: [w, x, y, z]
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_derivative(q, omega):
    # omega is a 3-vector; create quaternion [0, omega]
    omega_quat = np.concatenate(([0.0], omega))
    return 0.5 * quaternion_mult(q, omega_quat)

def normalize_quat(q):
    return q / np.linalg.norm(q)

def rotation_matrix(q):
    q = normalize_quat(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),       2 * (x*z + y*w)],
        [2 * (x*y + z*w),       1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),       1 - 2 * (x**2 + y**2)]
    ])
    return R

def euler_from_quaternion(q):
    w, x, y, z = q
    roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return roll, pitch, yaw

def quaternion_from_euler(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def skew(vec):
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def small_angle_quat(delta):
    angle = np.linalg.norm(delta)
    if angle < 1e-6:
        return np.array([1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]])
    else:
        axis = delta / angle
        return np.array([np.cos(angle / 2), axis[0] * np.sin(angle / 2),
                         axis[1] * np.sin(angle / 2), axis[2] * np.sin(angle / 2)])

#############################
# Propagation Functions
#############################
# State vector:
# x[0:4]   -> quaternion (q)
# x[4:7]   -> velocity (v)
# x[7:10]  -> position (p)
# x[10:13] -> gyro bias (b_g)
# x[13:16] -> accel bias (b_a)
# Gravity is fixed: g = [0, 0, -9.81]

def state_deriv(x, u):
    q   = x[0:4]
    v   = x[4:7]
    p   = x[7:10]
    b_g = x[10:13]
    b_a = x[13:16]
    
    # Corrected angular velocity:
    omega_corr = u['omega_meas'] - b_g
    q_dot = quat_derivative(q, omega_corr)
    
    # Corrected accelerometer measurement:
    a_corr = u['a_meas'] - b_a
    R_mat = rotation_matrix(q)
    g = np.array([0, 0, -9.81])
    v_dot = R_mat @ a_corr + g
    p_dot = v
    
    b_g_dot = np.zeros(3)
    b_a_dot = np.zeros(3)
    
    return np.concatenate([q_dot, v_dot, p_dot, b_g_dot, b_a_dot])

def rk4_step(x, u, dt):
    k1 = state_deriv(x, u)
    k2 = state_deriv(x + 0.5 * dt * k1, u)
    k3 = state_deriv(x + 0.5 * dt * k2, u)
    k4 = state_deriv(x + dt * k3, u)
    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[0:4] = normalize_quat(x_next[0:4])
    return x_next

#############################
# Chaotic Trajectory Generators
#############################

def chaotic_position(t, freq_x1=0.2, freq_x2=3.0,
                     freq_y1=0.1, freq_y2=2.3,
                     freq_z1=0.05, freq_z2=7.0):
    x = 8 * np.sin(freq_x1 * t) + 2 * np.sin(freq_x2 * t)
    y = 16 * np.sin(freq_y1 * t + 1.0) + 4 * np.sin(freq_y2 * t + 0.5)
    z = 50 + 40 * np.sin(freq_z1 * t) + 10 * np.sin(freq_z2 * t)
    return np.array([x, y, z])

def chaotic_velocity(t, freq_x1=0.2, freq_x2=3.0,
                     freq_y1=0.1, freq_y2=2.3,
                     freq_z1=0.05, freq_z2=7.0):
    vx = 8 * freq_x1 * np.cos(freq_x1 * t) + 2 * freq_x2 * np.cos(freq_x2 * t)
    vy = 16 * freq_y1 * np.cos(freq_y1 * t + 1.0) + 4 * freq_y2 * np.cos(freq_y2 * t + 0.5)
    vz = 40 * freq_z1 * np.cos(freq_z1 * t) + 10 * freq_z2 * np.cos(freq_z2 * t)
    return np.array([vx, vy, vz])

def chaotic_acceleration(t, freq_x1=0.2, freq_x2=3.0,
                         freq_y1=0.1, freq_y2=2.3,
                         freq_z1=0.05, freq_z2=7.0):
    ax = -8 * (freq_x1**2) * np.sin(freq_x1 * t) - 2 * (freq_x2**2) * np.sin(freq_x2 * t)
    ay = -16 * (freq_y1**2) * np.sin(freq_y1 * t + 1.0) - 4 * (freq_y2**2) * np.sin(freq_y2 * t + 0.5)
    az = -40 * (freq_z1**2) * np.sin(freq_z1 * t) - 10 * (freq_z2**2) * np.sin(freq_z2 * t)
    return np.array([ax, ay, az])

def chaotic_orientation(t, amp_roll=5.0, amp_pitch=5.0, amp_yaw=5.0,
                        freq_roll=0.2, freq_pitch=0.1, freq_yaw=0.05):
    roll  = amp_roll * np.sin(freq_roll * t)
    pitch = amp_pitch * np.sin(freq_pitch * t + 1.0)
    yaw   = amp_yaw * np.sin(freq_yaw * t)
    return np.array([roll, pitch, yaw])

def chaotic_angular_velocity(t, amp_roll=5.0, amp_pitch=5.0, amp_yaw=5.0,
                             freq_roll=0.2, freq_pitch=0.1, freq_yaw=0.05):
    roll_rate  = amp_roll * freq_roll * np.cos(freq_roll * t)
    pitch_rate = amp_pitch * freq_pitch * np.cos(freq_pitch * t + 1.0)
    yaw_rate   = amp_yaw * freq_yaw * np.cos(freq_yaw * t)
    return np.array([roll_rate, pitch_rate, yaw_rate])

#############################
# IMUSimulator Class (Redesigned ESKF)
#############################

class IMUSimulator:
    def __init__(self, dt=0.01, T_total=60.0, gps_rate=1.0, gps_delay=0.1,
                 gyro_noise_std=0.005, accel_noise_std=0.1, gps_noise_std=0.5,
                 seed=42,
                 # Dynamics parameters:
                 freq_x1=0.2, freq_x2=3.0,
                 freq_y1=0.1, freq_y2=2.3,
                 freq_z1=0.05, freq_z2=7.0,
                 amp_roll=5.0, amp_pitch=5.0, amp_yaw=5.0,
                 freq_roll=0.2, freq_pitch=0.1, freq_yaw=0.05):
        
        np.random.seed(seed)
        self.dt = dt
        self.T_total = T_total
        self.steps = int(T_total / dt)
        self.time = np.linspace(0, T_total, self.steps)
        
        # Fixed gravity
        self.g = np.array([0, 0, -9.81])
        self.gps_rate = gps_rate
        self.gps_update_steps = int(1.0 / (gps_rate * dt))
        self.gps_delay = gps_delay
        self.gps_delay_steps = int(gps_delay / dt)
        
        # Dynamics parameters
        self.freq_x1 = freq_x1; self.freq_x2 = freq_x2
        self.freq_y1 = freq_y1; self.freq_y2 = freq_y2
        self.freq_z1 = freq_z1; self.freq_z2 = freq_z2
        self.amp_roll = amp_roll; self.amp_pitch = amp_pitch; self.amp_yaw = amp_yaw
        self.freq_roll = freq_roll; self.freq_pitch = freq_pitch; self.freq_yaw = freq_yaw
        
        # Initial true state using chaotic functions
        euler0 = chaotic_orientation(0, amp_roll, amp_pitch, amp_yaw, freq_roll, freq_pitch, freq_yaw)
        self.q0 = quaternion_from_euler(euler0[0], euler0[1], euler0[2])
        self.v0 = chaotic_velocity(0, freq_x1, freq_x2, freq_y1, freq_y2, freq_z1, freq_z2)
        self.p0 = chaotic_position(0, freq_x1, freq_x2, freq_y1, freq_y2, freq_z1, freq_z2)
        b_g_true = np.zeros(3)
        b_a_true = np.zeros(3)
        self.x0_true = np.concatenate([self.q0, self.v0, self.p0, b_g_true, b_a_true])
        
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gps_noise_std = gps_noise_std
        
    def true_motion_profile(self, t):
        omega = chaotic_angular_velocity(t, self.amp_roll, self.amp_pitch, self.amp_yaw,
                                         self.freq_roll, self.freq_pitch, self.freq_yaw)
        a_inertial = chaotic_acceleration(t, self.freq_x1, self.freq_x2,
                                          self.freq_y1, self.freq_y2,
                                          self.freq_z1, self.freq_z2)
        return omega, a_inertial
    
    def true_a_body(self, x, t):
        q = x[0:4]
        R_mat = rotation_matrix(q)
        _, a_inertial = self.true_motion_profile(t)
        a_body = R_mat.T @ (a_inertial - self.g)
        return a_body
    
    def simulate_true_trajectory(self):
        n = 16
        x_true_hist = np.zeros((self.steps, n))
        x_true_hist[0] = self.x0_true
        for k in range(1, self.steps):
            t = self.time[k-1]
            omega, _ = self.true_motion_profile(t)
            u_true = {
                'omega_meas': omega,
                'a_meas': self.true_a_body(x_true_hist[k-1], t)
            }
            x_true_hist[k] = rk4_step(x_true_hist[k-1], u_true, self.dt)
        self.x_true_hist = x_true_hist
        return x_true_hist
    
    def simulate_measurements(self, gps_alpha=0.5):
        steps = self.steps
        time = self.time
        x_true_hist = self.x_true_hist
        
        z_gyro  = np.zeros((steps, 3))
        z_accel = np.zeros((steps, 3))
        z_gps   = np.full((steps, 3), np.nan)
        z_gps_filtered = np.full((steps, 3), np.nan)
        prev_filtered = None
        
        for k in range(steps):
            t = time[k]
            omega_true, _ = self.true_motion_profile(t)
            q_true = x_true_hist[k, 0:4]
            z_gyro[k] = omega_true + np.random.randn(3) * self.gyro_noise_std
            z_accel[k] = self.true_a_body(x_true_hist[k], t) + np.random.randn(3) * self.accel_noise_std
            if k % self.gps_update_steps == 0 and k >= self.gps_delay_steps:
                pos_true_delayed = x_true_hist[k - self.gps_delay_steps, 7:10]
                z_gps[k] = pos_true_delayed + np.random.randn(3) * self.gps_noise_std
                if prev_filtered is None:
                    filtered = z_gps[k]
                else:
                    filtered = gps_alpha * z_gps[k] + (1 - gps_alpha) * prev_filtered
                prev_filtered = filtered
                z_gps_filtered[k] = filtered
            else:
                z_gps[k] = np.nan
                z_gps_filtered[k] = np.nan
                
        self.z_gyro = z_gyro
        self.z_accel = z_accel
        self.z_gps = z_gps
        self.z_gps_filtered = z_gps_filtered
        return z_gyro, z_accel, z_gps, z_gps_filtered
    
    def run_ESKF(self, gains=None, zupt_enabled=True, zupt_vel_threshold=0.08, zupt_variance=1e-5):
        dt = self.dt
        steps = self.steps

        # Initialize state estimate: [q, v, p, b_g, b_a]
        x_hat = np.concatenate([self.q0, self.v0, self.p0, np.zeros(3), np.zeros(3)])
        n = 16
        x_est_hist = np.zeros((steps, n))
        x_est_hist[0] = x_hat

        d = 15  # error state dimension:
        P = np.eye(d) * 0.01

        # Process noise covariance Q with scaling from gains
        Q = np.zeros((d, d))
        Q[0:3, 0:3]   = (self.gyro_noise_std**2 * dt) * np.eye(3)
        Q[3:6, 3:6]   = (self.accel_noise_std**2 * dt) * np.eye(3)
        Q[6:9, 6:9]   = (0.01**2 * dt) * np.eye(3)
        Q[9:12, 9:12] = (1e-5 * dt) * np.eye(3)
        Q[12:15,12:15] = (1e-5 * dt) * np.eye(3)
        if gains is not None:
            Q[0:3, 0:3] *= gains['gyro_factor']
            Q[3:6, 3:6] *= gains['accel_factor']

        R_cov = (self.gps_noise_std**2) * np.eye(3)
        I_d = np.eye(d)

        # For measurement update filtering:
        prev_z = x_hat[7:10]  # initial "previous" measurement
        for k in range(1, steps):
            u = {
                'omega_meas': self.z_gyro[k-1],
                'a_meas': self.z_accel[k-1]
            }
            x_hat = rk4_step(x_hat, u, dt)

            # --- ZUPT Update (Zero Velocity Update) ---
            if zupt_enabled:
                vel_norm = np.linalg.norm(x_hat[4:7])
                if vel_norm < zupt_vel_threshold:
                    # Pseudo-measurement: velocity = 0
                    H_zupt = np.zeros((3, d))
                    H_zupt[:, 3:6] = np.eye(3)
                    R_zupt = zupt_variance * np.eye(3)
                    y_zupt = -x_hat[4:7]  # residual = 0 - vel_estimate
                    S_zupt = H_zupt @ P @ H_zupt.T + R_zupt
                    K_zupt = P @ H_zupt.T @ np.linalg.inv(S_zupt)
                    delta_x_zupt = K_zupt @ y_zupt

                    # Apply ZUPT correction to the state
                    delta_theta_zupt = delta_x_zupt[0:3]
                    delta_v_zupt     = delta_x_zupt[3:6]
                    delta_p_zupt     = delta_x_zupt[6:9]
                    delta_b_g_zupt   = delta_x_zupt[9:12]
                    delta_b_a_zupt   = delta_x_zupt[12:15]

                    dq_zupt = small_angle_quat(delta_theta_zupt)
                    x_hat[0:4] = normalize_quat(quaternion_mult(x_hat[0:4], dq_zupt))
                    x_hat[4:7] += delta_v_zupt
                    x_hat[7:10] += delta_p_zupt
                    x_hat[10:13] += delta_b_g_zupt
                    x_hat[13:16] += delta_b_a_zupt

                    P = (I_d - K_zupt @ H_zupt) @ P

            # Build F matrix (linearized error propagation)
            q = x_hat[0:4]
            R_mat = rotation_matrix(q)
            omega_meas = self.z_gyro[k-1]
            a_meas = self.z_accel[k-1]
            F = np.eye(d)
            # Attitude error propagation:
            F[0:3, 0:3] = np.eye(3) - skew(omega_meas) * dt
            # Velocity error propagation:
            F[3:6, 0:3] = -R_mat @ skew(a_meas) * dt
            F[3:6, 3:6] = np.eye(3)
            F[3:6, 12:15] = -R_mat * dt
            # Position error propagation:
            F[6:9, 3:6] = np.eye(3) * dt
            # Bias errors assumed constant.

            P = F @ P @ F.T + Q

            # Use filtered GPS measurement:
            # If filtered GPS is available, use it; otherwise, use a combination of predicted position and previous filtered.
            if not np.isnan(self.z_gps_filtered[k, 0]):
                z = self.z_gps_filtered[k]
                prev_z = z
            else:
                # Use a weighted blend between predicted position and previous measurement.
                z = (1 - gains['est_alpha']) * x_hat[7:10] + gains['est_alpha'] * prev_z
                prev_z = z

            y = z - x_hat[7:10]
            H = np.zeros((3, d))
            H[:, 6:9] = np.eye(3)
            S = H @ P @ H.T + R_cov
            K = P @ H.T @ np.linalg.inv(S)
            delta_x = K @ y

            delta_theta = delta_x[0:3]
            delta_v     = delta_x[3:6]
            delta_p     = delta_x[6:9]
            delta_b_g   = delta_x[9:12]
            delta_b_a   = delta_x[12:15]

            dq = small_angle_quat(delta_theta)
            x_hat[0:4] = normalize_quat(quaternion_mult(x_hat[0:4], dq))
            x_hat[4:7] += delta_v
            x_hat[7:10] += delta_p
            x_hat[10:13] += delta_b_g
            x_hat[13:16] += delta_b_a

            P = (I_d - K @ H) @ P

            x_est_hist[k] = x_hat

        self.x_est_hist = x_est_hist
        return x_est_hist
    
    def run_simulation(self, gains=None, plot_results=False, gps_alpha=0.5):
        self.simulate_true_trajectory()
        # In simulate_measurements, we pass gps_alpha so that it is tunable.
        self.simulate_measurements(gps_alpha=gps_alpha)
        self.run_ESKF(gains=gains)
        pos_true = self.x_true_hist[:, 7:10]
        pos_est = self.x_est_hist[:, 7:10]
        rmse = np.sqrt(np.mean(np.sum((pos_true - pos_est)**2, axis=1)))
        if plot_results:
            self.plot_results(rmse)
            self.error_analysis()
        return rmse
    
    def error_analysis(self):
        time = self.time
        pos_true = self.x_true_hist[:, 7:10]
        pos_est  = self.x_est_hist[:, 7:10]
        vel_true = self.x_true_hist[:, 4:7]
        vel_est  = self.x_est_hist[:, 4:7]
        pos_error = pos_true - pos_est
        vel_error = vel_true - vel_est
        pos_rmse = np.sqrt(np.mean(np.sum(pos_error**2, axis=1)))
        vel_rmse = np.sqrt(np.mean(np.sum(vel_error**2, axis=1)))
        print("Error Analysis:")
        print(f"Position RMSE: {pos_rmse:.3f} m")
        print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
        
        pos_error_norm = np.linalg.norm(pos_error, axis=1)
        plt.figure(figsize=(10, 4), dpi=600)
        plt.plot(self.time, pos_error_norm, label="Position Error Norm")
        plt.xlabel("Time (s)")
        plt.ylabel("Error Norm (m)")
        plt.title("Position Error Norm")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        vel_error_norm = np.linalg.norm(vel_error, axis=1)
        plt.figure(figsize=(10, 4), dpi=600)
        plt.plot(self.time, vel_error_norm, label="Velocity Error Norm")
        plt.xlabel("Time (s)")
        plt.ylabel("Error Norm (m/s)")
        plt.title("Velocity Error Norm")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8, 4), dpi=600)
        plt.hist(pos_error_norm, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel("Position Error Norm (m)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Position Error Norms")
        plt.grid(True)
        plt.show()
    
    def plot_results(self, rmse):
        time = self.time
        pos_true = self.x_true_hist[:, 7:10]
        pos_est  = self.x_est_hist[:, 7:10]
        vel_true = self.x_true_hist[:, 4:7]
        vel_est  = self.x_est_hist[:, 4:7]
        acc_true = np.diff(vel_true, axis=0) / self.dt
        acc_est  = np.diff(vel_est, axis=0) / self.dt
        time_acc = time[1:]
        
        # Position Plot with GPS overlay
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=600)
        # Plot X: True, Estimated and GPS (only non-NaN indices)
        gps_idx = ~np.isnan(self.z_gps[:, 0])
        
        axs[0].scatter(time[gps_idx], self.z_gps[gps_idx, 0], marker='o',
                       color='green', label="GPS X", s=20, alpha=0.3)
        axs[0].plot(time, pos_true[:, 0], label="True X")
        axs[0].plot(time, pos_est[:, 0], '--', label="Estimated X")
        axs[0].set_ylabel("X (m)")
        axs[0].legend()
        axs[0].set_title("Position X")
        
        
        axs[1].scatter(time[gps_idx], self.z_gps[gps_idx, 1], marker='o',
                       color='green', label="GPS Y", s=20, alpha=0.3)
        axs[1].plot(time, pos_true[:, 1], label="True Y")
        axs[1].plot(time, pos_est[:, 1], '--', label="Estimated Y")
        axs[1].set_ylabel("Y (m)")
        axs[1].legend()
        axs[1].set_title("Position Y")
        
        
        axs[2].scatter(time[gps_idx], self.z_gps[gps_idx, 2], marker='o',
                       color='green', label="GPS Z", s=20, alpha=0.3)
        axs[2].plot(time, pos_true[:, 2], label="True Z")
        axs[2].plot(time, pos_est[:, 2], '--', label="Estimated Z")
        axs[2].set_ylabel("Z (m)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].set_title("Position Z")
        fig.suptitle(f"True vs. Estimated Position (RMSE = {rmse:.3f} m)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # Plot Velocity
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=600)
        axs[0].plot(time, vel_true[:, 0], label="True Vx")
        axs[0].plot(time, vel_est[:, 0], '--', label="Estimated Vx")
        axs[0].set_ylabel("Vx (m/s)")
        axs[0].legend()
        axs[0].set_title("Velocity X")
        
        axs[1].plot(time, vel_true[:, 1], label="True Vy")
        axs[1].plot(time, vel_est[:, 1], '--', label="Estimated Vy")
        axs[1].set_ylabel("Vy (m/s)")
        axs[1].legend()
        axs[1].set_title("Velocity Y")
        
        axs[2].plot(time, vel_true[:, 2], label="True Vz")
        axs[2].plot(time, vel_est[:, 2], '--', label="Estimated Vz")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Vz (m/s)")
        axs[2].legend()
        axs[2].set_title("Velocity Z")
        fig.suptitle("True vs. Estimated Velocities", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # Plot Acceleration
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=600)
        axs[0].plot(time_acc, acc_true[:, 0], label="True Ax")
        axs[0].plot(time_acc, acc_est[:, 0], '--', label="Estimated Ax")
        axs[0].set_ylabel("Ax (m/s²)")
        axs[0].legend()
        axs[0].set_title("Acceleration X")
        
        axs[1].plot(time_acc, acc_true[:, 1], label="True Ay")
        axs[1].plot(time_acc, acc_est[:, 1], '--', label="Estimated Ay")
        axs[1].set_ylabel("Ay (m/s²)")
        axs[1].legend()
        axs[1].set_title("Acceleration Y")
        
        axs[2].plot(time_acc, acc_true[:, 2], label="True Az")
        axs[2].plot(time_acc, acc_est[:, 2], '--', label="Estimated Az")
        axs[2].set_ylabel("Az (m/s²)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].set_title("Acceleration Z")
        fig.suptitle("True vs. Estimated Accelerations", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # Plot Attitude (Euler Angles)
        true_euler = np.array([euler_from_quaternion(self.x_true_hist[k, 0:4]) for k in range(len(time))])
        est_euler  = np.array([euler_from_quaternion(self.x_est_hist[k, 0:4]) for k in range(len(time))])
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=600)
        axs[0].plot(time, true_euler[:, 0], label="True Roll")
        axs[0].plot(time, est_euler[:, 0], '--', label="Estimated Roll")
        axs[0].set_ylabel("Roll (rad)")
        axs[0].legend()
        axs[0].set_title("Roll")
        
        axs[1].plot(time, true_euler[:, 1], label="True Pitch")
        axs[1].plot(time, est_euler[:, 1], '--', label="Estimated Pitch")
        axs[1].set_ylabel("Pitch (rad)")
        axs[1].legend()
        axs[1].set_title("Pitch")
        
        axs[2].plot(time, true_euler[:, 2], label="True Yaw")
        axs[2].plot(time, est_euler[:, 2], '--', label="Estimated Yaw")
        axs[2].set_ylabel("Yaw (rad)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].set_title("Yaw")
        fig.suptitle("True vs. Estimated Attitude (Euler Angles)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # 3D Trajectory Plot using Plotly (with GPS data)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=pos_true[:, 0], y=pos_true[:, 1], z=pos_true[:, 2],
            mode='lines', name='True Trajectory', line=dict(width=3)
        ))
        fig3d.add_trace(go.Scatter3d(
            x=pos_est[:, 0], y=pos_est[:, 1], z=pos_est[:, 2],
            mode='lines', name='Estimated Trajectory', line=dict(width=3)
        ))
        # Add GPS data (only non-NaN measurements)
        gps_idx = ~np.isnan(self.z_gps[:, 0])
        gps_positions = self.z_gps[gps_idx]
        fig3d.add_trace(go.Scatter3d(
            x=gps_positions[:, 0], y=gps_positions[:, 1], z=gps_positions[:, 2],
            mode='markers', name='GPS Measurements', marker=dict(size=3)
        ))
        fig3d.update_layout(
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
            title=f"3D Trajectory (RMSE = {rmse:.3f} m) with GPS",
            margin=dict(l=0, r=0, b=0, t=50)
        )
        fig3d.show()

#############################
# Optuna Objective Function
#############################

def objective(trial, params):
    gains = {}
    gains['gyro_factor'] = trial.suggest_float('gyro_factor', 1e-3, 1e+1, log=True)
    gains['accel_factor'] = trial.suggest_float('accel_factor', 1e-3, 1e+1, log=True)
    gains['gps_alpha'] = trial.suggest_float('gps_alpha', 0.01, 0.99)
    gains['est_alpha'] = trial.suggest_float('est_alpha', 0.01, 0.99)
    
    simulator = IMUSimulator(
        dt=params['dt'],
        T_total=params['T_total'],
        gps_rate=params['gps_rate'],
        gps_delay=params['gps_delay'],
        gyro_noise_std=params['gyro_noise_std'],
        accel_noise_std=params['accel_noise_std'],
        gps_noise_std=params['gps_noise_std'],
        seed=params['seed'],
        freq_x1=params.get('freq_x1', 0.2),
        freq_x2=params.get('freq_x2', 3.0),
        freq_y1=params.get('freq_y1', 0.1),
        freq_y2=params.get('freq_y2', 2.3),
        freq_z1=params.get('freq_z1', 0.05),
        freq_z2=params.get('freq_z2', 7.0),
        amp_roll=params.get('amp_roll', 5.0),
        amp_pitch=params.get('amp_pitch', 5.0),
        amp_yaw=params.get('amp_yaw', 5.0),
        freq_roll=params.get('freq_roll', 0.2),
        freq_pitch=params.get('freq_pitch', 0.1),
        freq_yaw=params.get('freq_yaw', 0.05)
    )
    
    rmse = simulator.run_simulation(gains=gains, plot_results=False, gps_alpha=gains['gps_alpha'])
    return rmse

#############################
# Main
#############################

if __name__ == '__main__':
    params = {
        'dt': 0.01,
        'T_total': 60.0,
        'gps_rate': 10.0,
        'gps_delay': 0.1,
        'gyro_noise_std': 0.05,   
        'accel_noise_std': 0.002, 
        'gps_noise_std': 1.5,
        'seed': 42,
        'freq_x1': 0.2, 'freq_x2': 0.6,
        'freq_y1': 0.1, 'freq_y2': 1.2,
        'freq_z1': 0.05, 'freq_z2': 2.0,
        #'freq_z1': 0.0, 'freq_z2': 0.0,
        'amp_roll': 0.5,
        'amp_pitch': 1.0,
        'amp_yaw': 1.0,
        'freq_roll': 0.1,
        'freq_pitch': 0.1,
        'freq_yaw': 0.1
    }
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, params), n_trials=50)
    print("Best trial:")
    print(f"  RMSE: {study.best_value:.3f}")
    print(f"  Tuned Gains: {study.best_params}")
    
    simulator = IMUSimulator(
        dt=params['dt'],
        T_total=params['T_total'],
        gps_rate=params['gps_rate'],
        gps_delay=params['gps_delay'],
        gyro_noise_std=params['gyro_noise_std'],
        accel_noise_std=params['accel_noise_std'],
        gps_noise_std=params['gps_noise_std'],
        seed=params['seed'],
        freq_x1=params.get('freq_x1', 0.2),
        freq_x2=params.get('freq_x2', 3.0),
        freq_y1=params.get('freq_y1', 0.1),
        freq_y2=params.get('freq_y2', 2.3),
        freq_z1=params.get('freq_z1', 0.05),
        freq_z2=params.get('freq_z2', 7.0),
        amp_roll=params.get('amp_roll', 5.0),
        amp_pitch=params.get('amp_pitch', 5.0),
        amp_yaw=params.get('amp_yaw', 5.0),
        freq_roll=params.get('freq_roll', 0.2),
        freq_pitch=params.get('freq_pitch', 0.1),
        freq_yaw=params.get('freq_yaw', 0.05)
    )
    final_rmse = simulator.run_simulation(gains=study.best_params, plot_results=True, gps_alpha=study.best_params['gps_alpha'])
    print(f"Final RMSE with redesigned ESKF: {final_rmse:.3f}")
