import numpy as np

h=400 #km
R_mars = 3396.19 #km
r_lmo = R_mars + h
mu = 42828.3 #km^3/s^2
theta_lmo_rate = 0.000884797 #rad/s
mars_period_min = ((24*60) + 37)
mars_period_sec = mars_period_min*60
mars_period_hr = mars_period_min/60
gmo_period_sec = mars_period_sec
r_gmo = 20424.2 #km
theta_gmo_rate = 0.0000709003 #rad/s

#i_r points to s/c
#i_h direction of H
#i_theta = i_h x i_r
#antenna is in the direction of -b_1
#mars_sensor is in direction of +b_1
#solar panel normal is in direction of +b_3

sigma_bn_0 = np.array([0.3, -0.4, 0.5]) #MRPs
# sigma_bn_0_norm = sigma_bn_0/np.linalg.norm(sigma_bn_0)
omega_bn_0 = np.array([1.00, 1.75, -2.20]) #deg/s
I_b = np.array([[10, 0, 0],
               [0, 5, 0],
               [0, 0, 7.5]]) #kg*m^2
I_b_inv = np.linalg.inv(I_b)
# Assume s/c can create any 3d control torque vector u
# Sun is infinite distance away and always in the n_2 direction
# Detumble will point antenna at GMO mothership, sensor at Mars (nadir or -r), or solar panels at sun
# Any time s/c is on sunlit side (ie +n_2 position coordinate), solar panels need to point at sun
# This means b_3 will point in n_2
# To complete the 3D frame, assume r_1 must point in the -n_1 direction
# On shaded side (ie -n_2 position coordinate), must be in comm or sci mode
# In SCI mode, b_1 must point in nadir direction
# To complete frame, assume r_2 must line up with orbit alone track axis i_theta
# In COMM mode, LMO and GMO position vectors have angular difference of <35 degrees
# This means -b_1 must point in direction of GMO

# we will use solve_ivp to ensure use of RK45


# Initial state
X_0 = [sigma_bn_0, omega_bn_0]  # Initial conditions
tmax = 6500  # Set the value of tmax
dt = .1  # Set the value of Δt
t_0 = 0.0  # Initial time

# # Function to evaluate current reference frame states (example, replace with actual evaluation)
# def evaluate_reference_frame(tn):
#     # Replace with actual logic to compute RN(t), NωR/N(t), etc.
#     return RN, omega_rn

# # Function to calculate control tracking errors (example)
# def control_tracking_errors():
#     # Replace with actual logic to compute σB/R and BωB/R
#     return sigma_br, omega_br

# # Function to determine control solution (example)
# def control_solution():
#     # Replace with actual logic to determine control solution u
#     return u

# # Function to compute the differential (replace with your differential equation model)
# def f(Xn, t_n, u):
#     # Replace with actual system dynamics function for f(Xn, tn, u)
#     return Xn

# # Run the time-stepping loop
# X_n = X_0  # Initialize the state
# t_n = t_0
# sigma_bn = sigma_bn_0
# omega_bn = omega_bn_0
# while t_n < tmax:
#     if new_control_required():
#         # If new control is required, evaluate reference frame states and control tracking errors
#         RN_t, NωR_N_t = evaluate_reference_frame(tn)
#         σB_R, BωB_R = control_tracking_errors()
#         u = control_solution()

#     # Run the 4th order Runge-Kutta integration
#     k1 = dt * f(X_n, t_n, u)
#     k2 = dt * f([X_n + k1/2], t_n + dt/2, u)
#     k3 = dt * f([X_n + k2/2], t_n + dt/2, u)
#     k4 = dt * f([X_n + k3], t_n + dt, u)
#     X_n = [X_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4)] # Update the state

#     # Check for control error and map to shadow set if necessary
#     if abs(sigma_bn) > 1:
#         # Map σB/N to shadow set
#         pass

#     # Update time and save states
#     tn += Δt
    # Save spacecraft states Xn and u (you can save this to a file or list as needed)
    # save_states(Xn, u)

# Project Tasks

# Task 1: Orbit Simulation (5 points)
# pos = r*i_r
# Derive inertial s/c velocity r_dot. Note that for circular orbits, theta_dot is constant
# Write a function whose inputs are radius r and 313 angles omega, i, theta, and outputs are the inertial pos vector N_r and vel N_r_dot
lmo_omega_0 = np.deg2rad(20)
lmo_i_0 = np.deg2rad(30)
lmo_theta_0 = np.deg2rad(60) #function of time
gmo_omega_0 = 0
gmo_i_0 = 0
gmo_theta_0 = np.deg2rad(250) #function of time


def theta_lmo(t):
    return lmo_theta_0 + t*theta_lmo_rate

def theta_gmo(t):
    return gmo_theta_0 + t*theta_gmo_rate

def s(theta):
    return np.sin(theta)
def c(theta):  
    return np.cos(theta)
def Euler313toDCM(t1, t2, t3):
    # Convert 313 angles to DCM - From appendix B.1
    return np.array([[c(t3)*c(t1)-s(t3)*c(t2)*s(t1), c(t3)*s(t1)+s(t3)*c(t2)*c(t1), s(t3)*s(t2)],
                     [-s(t3)*c(t1)-c(t3)*c(t2)*s(t1), -s(t3)*s(t1)+c(t3)*c(t2)*c(t1), c(t3)*s(t2)],
                     [s(t2)*s(t1), -s(t2)*c(t1), c(t2)]])
    
# Calculate the inertial position vector N_r and velocity N_r_dot
def orbit_sim(r, omega, i, theta):
    # O : {i_r, i_theta, i_h}
    # N : {n_1, n_2, n_3}
    NO = Euler313toDCM(omega, i, theta)
    # Convert direction of i_r to N
    N_r = NO @ np.array([r, 0, 0])
    # Convert direction of i_theta to N
    N_r_dot = NO @ np.array([0, r * np.sqrt(mu/r**3), 0])
    return N_r, N_r_dot


# confirm the operation by checking orbit_sim(r_lmo, omega_lmo, i_lmo, theta_lmo(450)) and orbit_sim(r_gmo, omega_gmo, i_gmo, theta_gmo(1150))
N_r_lmo, N_r_lmo_dot = orbit_sim(r_lmo, lmo_omega_0, lmo_i_0, theta_lmo(450))
print(np.sqrt(mu/r_lmo**3))
print(np.sqrt(mu/r_gmo**3))

print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
print("N_r_lmo = ", N_r_lmo)
print("N_r_lmo_dot = ", N_r_lmo_dot)
N_r_gmo, N_r_gmo_dot = orbit_sim(r_gmo, gmo_omega_0, gmo_i_0, theta_gmo(1150))
print("N_r_gmo = ", N_r_gmo)
print("N_r_gmo_dot = ", N_r_gmo_dot)
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
# Task 2: Orbit Frame Orientation (5 points)
