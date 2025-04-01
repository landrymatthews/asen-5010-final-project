import numpy as np
from sympy import symbols, pprint, cos, sin, Matrix, pi
import sympy as sp
import matplotlib.pyplot as plt

# i_r points to s/c
# i_h direction of H
# i_theta = i_h x i_r
# antenna is in the direction of -b_1
# mars_sensor is in direction of +b_1
# solar panel normal is in direction of +b_3

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


############################## Welcome ##############################
print("Welcome to the ASEN 5010 Capstone Project")
print("Attitude Dynamics and Control of a Nano-Satellite Orbiting Mars")

############################## Initial state ##############################
h = 400  # km
R_mars = 3396.19  # km
r_lmo = R_mars + h
mu = 42828.3  # km^3/s^2
theta_lmo_rate = 0.000884797  # rad/s
mars_period_min = (24 * 60) + 37
mars_period_sec = mars_period_min * 60
mars_period_hr = mars_period_min / 60
gmo_period_sec = mars_period_sec
r_gmo = 20424.2  # km
theta_gmo_rate = 0.0000709003  # rad/s
tmax = 6500  # Set the value of tmax
dt = 0.1  # Set the value of Δt
t_0 = 0.0  # Initial time
sigma_bn_0 = np.array([0.3, -0.4, 0.5])  # MRPs
omega_bn_0 = np.array([1.00, 1.75, -2.20])  # deg/s
I_b = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 7.5]])  # kg*m^2
I_b_inv = np.linalg.inv(I_b)
X_0 = np.concatenate((sigma_bn_0, omega_bn_0))  # Initial conditions
omega_lmo = np.deg2rad(20)
i_lmo = np.deg2rad(30)
theta_lmo_0 = np.deg2rad(60)  # function of time
omega_gmo = 0
i_gmo = 0
theta_gmo_0 = np.deg2rad(250)  # function of time


############################## Helper Functions ##############################
def writeToFile(path, data):
    str_to_write = ""
    with open(path, "w+") as file:
        if data.ndim == 2:
            for row in data:
                for element in row:
                    str_to_write += str(element) + " "
        else:
            for element in data:
                str_to_write += str(element) + " "
        file.write(str_to_write.rstrip())


def theta_lmo(t):
    return theta_lmo_0 + t * theta_lmo_rate


def theta_gmo(t):
    return theta_gmo_0 + t * theta_gmo_rate


def s(theta):
    return np.sin(theta)


def c(theta):
    return np.cos(theta)

# Returns a skew-symmetric matrix from a vector
def tilde(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Euler313toDCM(t1, t2, t3):
    # Convert 313 angles to DCM - From appendix B.1
    return np.array(
        [
            [
                c(t3) * c(t1) - s(t3) * c(t2) * s(t1),
                c(t3) * s(t1) + s(t3) * c(t2) * c(t1),
                s(t3) * s(t2),
            ],
            [
                -s(t3) * c(t1) - c(t3) * c(t2) * s(t1),
                -s(t3) * s(t1) + c(t3) * c(t2) * c(t1),
                c(t3) * s(t2),
            ],
            [s(t2) * s(t1), -s(t2) * c(t1), c(t2)],
        ]
    )


def DCM2Quaternion(C):
    # Initialize stuff
    B = np.zeros(4)  # B^2 array
    b = np.zeros(4)  # resulting set of quaternions
    trc = np.trace(C)

    B[0] = (1 + trc) / 4
    B[1] = (1 + 2 * C[0, 0] - trc) / 4
    B[2] = (1 + 2 * C[1, 1] - trc) / 4
    B[3] = (1 + 2 * C[2, 2] - trc) / 4

    # Find the index of the maximum value in B2
    i = np.argmax(B)

    # Calculate quaternion based on sheppard's method
    if i == 0:
        b[0] = np.sqrt(B[0])
        b[1] = (C[1, 2] - C[2, 1]) / (4 * b[0])
        b[2] = (C[2, 0] - C[0, 2]) / (4 * b[0])
        b[3] = (C[0, 1] - C[1, 0]) / (4 * b[0])
    elif i == 1:
        b[1] = np.sqrt(B[1])
        b[0] = (C[1, 2] - C[2, 1]) / (4 * b[1])
        if b[0] < 0:
            b[1] = -b[1]
            b[0] = -b[0]
        b[2] = (C[0, 1] + C[1, 0]) / (4 * b[1])
        b[3] = (C[2, 0] + C[0, 2]) / (4 * b[1])
    elif i == 2:
        b[2] = np.sqrt(B[2])
        b[0] = (C[2, 0] - C[0, 2]) / (4 * b[2])
        if b[0] < 0:
            b[2] = -b[2]
            b[0] = -b[0]
        b[1] = (C[0, 1] + C[1, 0]) / (4 * b[2])
        b[3] = (C[1, 2] + C[2, 1]) / (4 * b[2])
    elif i == 3:
        b[3] = np.sqrt(B[3])
        b[0] = (C[0, 1] - C[1, 0]) / (4 * b[3])
        if b[0] < 0:
            b[3] = -b[3]
            b[0] = -b[0]
        b[1] = (C[2, 0] + C[0, 2]) / (4 * b[3])
        b[2] = (C[1, 2] + C[2, 1]) / (4 * b[3])

    return b


def MRP2DCM(sigma):
    tilde_sigma = tilde(sigma)
    return np.eye(3) + (
        8 * tilde_sigma @ tilde_sigma - 4 * (1 - sigma @ sigma) * tilde_sigma
    ) / ((1 + sigma @ sigma) ** 2)

def checkShadowSet(sigma):
    if (np.linalg.norm(sigma) > 1):
        return -sigma / (sigma @ sigma)
    else:
        return sigma
    
def DCM2MRP(C):
    b = DCM2Quaternion(C)
    divisor = 1 + b[0]
    sigma = np.array([b[1], b[2], b[3]]) / divisor
    return checkShadowSet(sigma)


############################## Task 1: Orbit Simulation (5 points) ##############################
# pos = r*i_r
# Derive inertial s/c velocity r_dot.
# Note that for circular orbits, theta_dot is constant
print("\n\nBEGIN TASK 1")


# Write a function whose inputs are radius r and 313 angles omega, i, theta,
# and outputs are the inertial pos vector N_r and vel N_r_dot
# Calculate the inertial position vector N_r and velocity N_r_dot
def orbit_sim(r, omega, i, theta):
    # O : {i_r, i_theta, i_h} aka H frame
    # N : {n_1, n_2, n_3}
    ON = Euler313toDCM(omega, i, theta)
    NO = ON.T
    # Convert direction of i_r to N
    N_r = NO @ np.array([r, 0, 0])
    # Convert direction of i_theta to N
    if r == r_lmo:
        N_r_dot = NO @ np.array([0, r * theta_lmo_rate, 0])
    if r == r_gmo:
        N_r_dot = NO @ np.array([0, r * theta_gmo_rate, 0])
    return N_r, N_r_dot


# confirm the operation by checking orbit_sim(r_lmo, omega_lmo, i_lmo, theta_lmo(450))
# and orbit_sim(r_gmo, omega_gmo, i_gmo, theta_gmo(1150))
N_r_lmo, N_r_lmo_dot = orbit_sim(r_lmo, omega_lmo, i_lmo, theta_lmo(450))
N_r_gmo, N_r_gmo_dot = orbit_sim(r_gmo, omega_gmo, i_gmo, theta_gmo(1150))
print("rLMO = ", N_r_lmo)
print("vLMO = ", N_r_lmo_dot)
print("rGMO = ", N_r_gmo)
print("vGMO = ", N_r_gmo_dot)
writeToFile("./tasks/task 1/rLMO.txt", N_r_lmo)
writeToFile("./tasks/task 1/vLMO.txt", N_r_lmo_dot)
writeToFile("./tasks/task 1/rGMO.txt", N_r_gmo)
writeToFile("./tasks/task 1/vGMO.txt", N_r_gmo_dot)



############################## Task 2: Orbit Frame Orientation (5 points) ##############################
print("\n\nBEGIN TASK 2")
Omega, i = symbols("Omega i")
t = symbols("t", positive=True)
theta = sp.Function(symbols("theta"))
HN = Matrix(
    [
        [
            cos(theta(t)) * cos(Omega) - sin(theta(t)) * cos(i) * sin(Omega),
            cos(theta(t)) * sin(Omega) + sin(theta(t)) * cos(i) * cos(Omega),
            sin(theta(t)) * sin(i),
        ],
        [
            -sin(theta(t)) * cos(Omega) - cos(theta(t)) * cos(i) * sin(Omega),
            -sin(theta(t)) * sin(Omega) + cos(theta(t)) * cos(i) * cos(Omega),
            cos(theta(t)) * sin(i),
        ],
        [sin(i) * sin(Omega), -sin(i) * cos(Omega), cos(i)],
    ]
)
with open('./latex/task_2_HN.tex', "w+") as file:
    file.write((sp.latex(HN)))

# Write a function whose input is time t and output is DCM HN(t) for the LMO
def getHNforLMO(t):
    return Euler313toDCM(omega_lmo, i_lmo, theta_lmo(t))


# Validate the operation by computing HN(300)
t = 300  # sec
HN_at_t = getHNforLMO(t)
print("HN(t = " + str(t) + "s) = ", HN_at_t)
writeToFile("./tasks/task 2/HN.txt", HN_at_t)


############################## Task 3: Sun-Pointing Reference Frame Orientation (10 points) ##############################
print("\n\nBEGIN TASK 3")
# First determine and analytic expression for Rs by defining DCM [RsN]
RsN = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
with open('./latex/RsN.tex', "w+") as file:
    file.write((sp.latex(RsN)))

# Write a function that returns RsN
def getRsN():
    return RsN

def getOmegaRsN():
    return np.array([0, 0, 0])


# Validate the evalutation of RsN by providing numerical values for t=0s
print("RsN(t = 0s) = ", getRsN())
writeToFile("./tasks/task 3/RsN.txt", getRsN())

# Angular velocity is [0, 0, 0] since the DCM is not a function of time
print("N_ω_Rn/N = ", getOmegaRsN())
writeToFile("./tasks/task 3/omega_rs_n.txt", getOmegaRsN())


############################## Task 4: Nadir-Pointing Reference Frame Orientation (10 points) ##############################
print("\n\nBEGIN TASK 4")
# First determine and analytic expression for Rn by defining DCM [RnN]
RnH = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
with open('./latex/RnH.tex', "w+") as file:
    file.write((sp.latex(RnH)))
with open('./latex/RnN.tex', "w+") as file:
    file.write((sp.latex(RnH @ HN)))


# Write a function that returns RnN
def getRnN(t):
    return RnH @ getHNforLMO(t)


# Write a function that determines angular velocity vector omega_rn_n
def getOmegaRnN(t):
    NRn = getRnN(t).T
    return NRn @ [0, 0, -theta_lmo_rate]  # ω = θ_dot*i_h = -θ_dot*r_3


t = 330  # sec
# Validate the evalutation of RnN by providing numerical values for t = 330s
RnN_at_t = getRnN(t)
print("RnN(t = " + str(t) + "s) = ", RnN_at_t)
writeToFile("./tasks/task 4/RnN.txt", RnN_at_t)

# What is the angular velocity @ t = 330s
omega_rnn_at_t = getOmegaRnN(t)
print("N_ω_Rn/N(t = " + str(t) + "s) = ", omega_rnn_at_t)
writeToFile("./tasks/task 4/omega_rn_n.txt", omega_rnn_at_t)


############################## Task 5: GMO-Pointing Reference Frame Orientation (10 points) ##############################
print("\n\nBEGIN TASK 5")


# dr = r_gmo - r_lmo  => lets get these in N frame to make cross product easy
# H : {i_r, i_theta, i_h} - Note theres one for GMO, one for LMO, depending on theta
# N : {n_1, n_2, n_3}
# Write a function that returns RcN
def getRcNExpr():
    NH = HN.T
    theta_gmo_expr = sp.Function(symbols("theta_GMO"))
    theta_lmo_expr = sp.Function(symbols("theta_LMO"))
    NH_gmo = NH.subs([(Omega, omega_gmo), (i, i_gmo), (theta, theta_gmo_expr)])
    NH_lmo = NH.subs([(Omega, omega_lmo), (i, i_lmo), (theta, theta_lmo_expr)])
    N_r_gmo_col = NH_gmo @ sp.Matrix([r_gmo, 0, 0])
    N_r_lmo_col = NH_lmo @ sp.Matrix([r_lmo, 0, 0])

    N_dr_col = N_r_gmo_col - N_r_lmo_col
    N_r1_col = -N_dr_col.normalized()
    N_r2_col = (N_dr_col.cross(sp.Matrix([0, 0, 1]))).normalized()
    N_r3_col = N_r1_col.cross(N_r2_col).normalized()

    return (sp.Matrix.hstack(N_r1_col, N_r2_col, N_r3_col)).T


def getRcN(t):
    RcN = getRcNExpr()
    RcN_f = sp.lambdify(
        ["t"],
        RcN,
        modules=["numpy", {"theta_GMO": theta_gmo}, {"theta_LMO": theta_lmo}],
    )
    return RcN_f(t)


# Write a function that determines angular velocity vector omega_rc_n
def getOmegaRcNAnalytically(time):
    t = symbols("t", positive=True)
    RcN = getRcNExpr()
    # RcNT = RcN.T

    # Replace with base functions so sympy knows how to derive wrt t
    theta_gmo_expr = sp.Function(symbols("theta_GMO"))
    theta_lmo_expr = sp.Function(symbols("theta_LMO"))
    replace_gmo = theta_gmo_0 + t * theta_gmo_rate
    replace_lmo = theta_lmo_0 + t * theta_lmo_rate
    replace_dict = {theta_gmo_expr(t): replace_gmo, theta_lmo_expr(t): replace_lmo}
    RcN_rep = RcN.subs(replace_dict)

    # Now sympy can take derivative wrt t
    RcN_dot = sp.diff(RcN_rep, t)
    omega_tilde = -RcN_dot @ RcN.T
    # with open('./latex/insane_matrix.tex', "w+") as file:
    #     file.write((sp.latex(omega_tilde)))
    omega_tilde_f = sp.lambdify(
        ["t"],
        omega_tilde,
        modules=["numpy", {"theta_GMO": theta_gmo}, {"theta_LMO": theta_lmo}],
    )

    ssm = omega_tilde_f(time)
    ssm = (ssm - ssm.T) / 2 # Force diagonals to 0
    R_omega_rcn = sp.Matrix([-ssm[1, 2], ssm[0, 2], -ssm[0, 1]])
    N_omega_rcn = RcN.T @ R_omega_rcn
    N_omega_rcn_f = sp.lambdify(
        ["t"],
        N_omega_rcn,
        modules=["numpy", {"theta_GMO": theta_gmo}, {"theta_LMO": theta_lmo}],
    )
    N_omega_rcn_real = N_omega_rcn_f(time)
    return N_omega_rcn_real.flatten()

def getOmegaRcN(time):
    dt = 1e-6

    RcN_t = getRcN(time)
    RcN_plus = getRcN(time + dt)
    RcN_minus = getRcN(time - dt)
    RcN_dot = (RcN_plus - RcN_minus) / (2 * dt)

    ssm = -RcN_dot @ RcN_t.T
    ssm = (ssm + ssm) / 2 # Force diagonals to 0
    R_omega_rcn = np.array([-ssm[1, 2], ssm[0, 2], -ssm[0, 1]])
    N_omega_rcn = RcN_t.T @ R_omega_rcn
    return N_omega_rcn


t = 330
RcN_at_t = getRcN(t)
omega_RcN_num_at_t = getOmegaRcN(t)
omega_RcN_anal_at_t = getOmegaRcNAnalytically(t)
print("RcN = ", RcN_at_t)
print("Numerical ω = ", omega_RcN_num_at_t)
print("Analytical ω = ", omega_RcN_anal_at_t)
writeToFile("./tasks/task 5/RcN.txt", RcN_at_t)
writeToFile("./tasks/task 5/omega_rc_n_num.txt", omega_RcN_num_at_t)
writeToFile("./tasks/task 5/omega_rc_n_anal.txt", omega_RcN_anal_at_t)


############################## Task 6: Attitude Error Evaluation (10 points) ##############################
print("\n\nBEGIN TASK 6")

# Write function that returns tracking errors sigma_br and omega_br
def getTrackingErrors(t, sigma_bn, B_omega_bn, RN, N_omega_rn):
    # Get σ_BR from σ_BN and RN DCM
    BN = MRP2DCM(sigma_bn)
    BR = BN @ (RN.T)
    sigma_br = DCM2MRP(BR)

    # Get ω_br from ω_bn and ω_rn
    print("\nheres BN: ", BN)
    print("heres B_omega_bn: ", B_omega_bn)
    print("heres N_omega_rn: ", N_omega_rn)
    print("heres B_omega_rn: ", BN @ N_omega_rn)
    print("\n")
    B_omega_br = B_omega_bn - (BN @ N_omega_rn)

    return sigma_br, B_omega_br


# Sun-pointing
t = 0
sigma, omega = getTrackingErrors(t, sigma_bn_0, omega_bn_0, getRsN(), getOmegaRsN())
print("Sun-Pointing Orientation")
print("σ_B/R = ", sigma)
print("ω_B/R = ", omega)
writeToFile("./tasks/task 6/sun-sigma.txt", sigma)
writeToFile("./tasks/task 6/sun-omega.txt", omega)

# Nadir-pointing
sigma, omega = getTrackingErrors(t, sigma_bn_0, omega_bn_0, getRnN(t), getOmegaRnN(t))
print("Nadir-Pointing Orientation")
print("σ_B/R = ", sigma)
print("ω_B/R = ", omega)
writeToFile("./tasks/task 6/nad-sigma.txt", sigma)
writeToFile("./tasks/task 6/nad-omega.txt", omega)

# GMO-pointing
sigma, omega = getTrackingErrors(t, sigma_bn_0, omega_bn_0, getRcN(t), getOmegaRcN(t))
print("GMO-Pointing Orientation")
print("σ_B/R = ", sigma)
print("ω_B/R = ", omega)
writeToFile("./tasks/task 6/gmo-sigma.txt", sigma)
writeToFile("./tasks/task 6/gmo-omega.txt", omega)




############################## Numerical Attitude Simulator (10 points) ##############################
print("\n\nBEGIN TASK 7")

# Write your own numerical integrator using RK45

# Constants for inertia matrix (for simplicity, assuming a diagonal matrix)

# Define the spacecraft dynamics (equation of motion)
# def dynamics(X, u):
#     sigma_BN = X[:3]  # MRP attitude
#     omega_BN = X[3:]  # Angular velocity
#     omega_BN_skew = tilde(omega_BN)
    
#     # Equation of motion
#     d_omega_BN = -np.dot(omega_BN_skew, I_b @ omega_BN) + u
#     return np.concatenate((d_sigma_BN(sigma_BN, omega_BN), d_omega_BN))

# # MRP kinematics
# def d_sigma_BN(sigma_BN, omega_BN):
#     # MRP update equations (for simplicity assume small angles)
#     sigma_dot = 0.5 * (np.eye(3) - tilde(sigma_BN)) @ omega_BN
#     return sigma_dot

# # Runge-Kutta 4th order integrator
# def rk4_integrator(f, X, u, dt):
#     k1 = f(X, u)
#     k2 = f(X + 0.5 * dt * k1, u)
#     k3 = f(X + 0.5 * dt * k2, u)
#     k4 = f(X + dt * k3, u)
#     return X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# # Initial conditions (assuming initial angular velocity and MRP are zero)

# # Time settings
# dt = 1e-1  # 1 second time step
# t_final = 100.0  # Total time for the integration
# time_steps = int(t_final / dt)

# # Control torque (zero initially)
# u = np.zeros(3)  # Control torque vector

# # Arrays to store results for plotting
# sigma_BN_history = []
# omega_BN_history = []
# T_history = []
# H_history = []

# # Integration loop (u = 0 for this part)
# X = X_0
# print(X)
# for t in range(time_steps):
#     sigma_BN_history.append(X[:3])
#     omega_BN_history.append(X[3:])
    
#     # Compute rotational kinetic energy T
#     print(X[3:])
#     T = 0.5 * np.dot(X[3:], I_b @ X[3:])
#     T_history.append(T)
    
#     # Compute angular momentum H
#     H = I_b @ X[3:]
#     H_history.append(H)
    
#     # Update attitude using RK4
#     X = rk4_integrator(dynamics, X, u, dt)

# # Results at 500 seconds
# sigma_BN_500 = sigma_BN_history[-1]
# omega_BN_500 = omega_BN_history[-1]
# T_500 = T_history[-1]
# H_500 = H_history[-1]

# print(f"MRP attitude at 500s: {sigma_BN_500}")
# print(f"Angular velocity at 500s: {omega_BN_500}")
# print(f"Rotational kinetic energy at 500s: {T_500}")
# print(f"Angular momentum at 500s: {H_500}")

# # Now apply control torque u = (0.01, -0.01, 0.02) Nm and integrate again for 100s
# u_fixed = np.array([0.01, -0.01, 0.02])  # Fixed control torque
# X = X_0  # Reset initial conditions
# sigma_BN_100_history = []

# # Run integration with control torque for 100 seconds
# for t in range(int(100 / dt)):
#     sigma_BN_100_history.append(X[:3])
    
#     # Update attitude using RK4
#     X = rk4_integrator(dynamics, X, u_fixed, dt)

# sigma_BN_100 = sigma_BN_100_history[-1]
# print(f"MRP attitude at 100s with control torque: {sigma_BN_100}")

# # Plot the results for visualization
# sigma_BN_history = np.array(sigma_BN_history)
# omega_BN_history = np.array(omega_BN_history)

# plt.subplots(figsize=(12, 6))

# # Plot MRP attitude history
# plt.subplot(2, 1, 1)
# plt.plot(np.linspace(0, t_final, len(sigma_BN_history)), sigma_BN_history[:, 0], label=r'$\sigma_1$')
# plt.plot(np.linspace(0, t_final, len(sigma_BN_history)), sigma_BN_history[:,1], label=r'$\sigma_2$')
# plt.plot(np.linspace(0, t_final, len(sigma_BN_history)), sigma_BN_history[:,2], label=r'$\sigma_3$')
# plt.title("MRP Attitude over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("MRP Components")
# plt.legend()

# # Plot Angular velocity history
# plt.subplot(2, 1, 2)
# plt.plot(np.linspace(0, t_final, len(omega_BN_history)), omega_BN_history[:,0], label=r'$\omega_1$')
# plt.plot(np.linspace(0, t_final, len(omega_BN_history)), omega_BN_history[:,1], label=r'$\omega_1$')
# plt.plot(np.linspace(0, t_final, len(omega_BN_history)), omega_BN_history[:,2], label=r'$\omega_1$')

# plt.title("Angular Velocity over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.legend()

# plt.tight_layout()
# plt.show()
