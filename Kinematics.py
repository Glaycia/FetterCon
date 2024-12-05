from casadi import MX, pi, sin, cos, vertcat, Opti
import casadi as np
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

base_attachment_poses = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 0.5, 0],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0.5, 1]
]
handle_attachment_poses = [
    [0.02, 0.02, 0.02],
    [-0.02, 0.02, 0.02],
    [0.02, 0, 0.02],
    [0.02, -0.02, -0.02],
    [-0.02, -0.02, -0.02],
    [0.02, 0, -0.02],
]
def rotationX(angle):
    RXM = MX.eye(4)
    RXM[1, 1] = cos(angle)
    RXM[1, 2] = -sin(angle)
    RXM[2, 1] = sin(angle)
    RXM[2, 2] = cos(angle)
    return RXM
def rotationY(angle):
    RYM = MX.eye(4)
    RYM[0, 0] = cos(angle)
    RYM[0, 2] = sin(angle)
    RYM[2, 0] = -sin(angle)
    RYM[2, 2] = cos(angle)
    return RYM
def rotationZ(angle):
    RZM = MX.eye(4)
    RZM[0, 0] = cos(angle)
    RZM[0, 1] = sin(angle)
    RZM[1, 0] = -sin(angle)
    RZM[1, 1] = cos(angle)
    return RZM
def forward_kinematics(T, RX, RY, RZ):
    # Rotation matrices
    RXM = rotationX(RX)
    RYM = rotationY(RY)
    RZM = rotationZ(RZ)

    # Transformation matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T

    # Calculate cable lengths
    lengths = []
    for i in range(len(base_attachment_poses)):
        # Transform handle attachment points to world coordinates
        homogenous_handle = vertcat(MX(handle_attachment_poses[i]), 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z

        # Calculate cable length as distance between base and transformed handle point
        length = ((transformed_handle_pos - MX(base_attachment_poses[i])).T @ 
                  (transformed_handle_pos - MX(base_attachment_poses[i])))**0.5
        lengths.append(length)
    # evaluated_lengths = [length.evalf() for length in lengths]
    return lengths

def inverse_kinematics(L):
    problem = Opti()
    # Variables
    T = problem.variable(3)  # Translation vector
    RX = problem.variable()  # Rotation around X
    RY = problem.variable()  # Rotation around Y
    RZ = problem.variable()  # Rotation around Z

    # Rotation Matrices
    RXM = rotationX(RX)
    RYM = rotationY(RY)
    RZM = rotationZ(RZ)

    # Full Transformation Matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T  # Add translation

    # Error Function
    error = 0
    for i in range(len(base_attachment_poses)):
        # Compute displacement vector
        homogenous_handle = vertcat(MX(handle_attachment_poses[i]), 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z
        displ_vec = transformed_handle_pos - MX(base_attachment_poses[i])

        # Add squared error
        error += (displ_vec.T @ displ_vec - L[i]**2)**2

    # Objective
    problem.minimize(error)

    # Solve
    problem.solver('ipopt')
    solution = problem.solve()

    # Return results
    return {
        "T": solution.value(T),
        "RX": solution.value(RX),
        "RY": solution.value(RY),
        "RZ": solution.value(RZ),
        "Error": solution.value(error)
    }

def force_kinematics(T, RX, RY, RZ, Force, Torque, max_tension, min_tension):
    # Rotation matrices
    RXM = rotationX(RX)
    RYM = rotationY(RY)
    RZM = rotationZ(RZ)

    # Transformation matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T

    # Calculate unit vectors
    unit_vec = []
    for i in range(len(base_attachment_poses)):
        # Transform handle attachment points to world coordinates
        homogenous_handle = vertcat(MX(handle_attachment_poses[i]), 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z

        # Calculate cable length as distance between base and transformed handle point
        delta = transformed_handle_pos - MX(base_attachment_poses[i])
        length = (delta.T @ delta)**0.5
        dir = delta/length
        unit_vec.append(dir)
    #minimize error of forces torques and minimize tensions
    problem = Opti()
    tensions = problem.variable(len(base_attachment_poses))

    solver_F = MX.zeros(3)
    solver_T = MX.zeros(3)

    for i in range(len(base_attachment_poses)):
        force_vec = unit_vec[i] * tensions[i]
        solver_F += force_vec
        torque_vec = np.cross(MX(handle_attachment_poses[i]), force_vec)
        solver_T += torque_vec
    
    error_f = (Force-solver_F)
    error_t = (Torque-solver_T)
    cost = error_f.T@error_f + error_t.T@error_t
    for i in range(len(base_attachment_poses)):
        problem.subject_to(tensions[i] <= max_tension)
        problem.subject_to(tensions[i] >= min_tension)

    problem.minimize(cost)
    problem.solver('ipopt')
    solution = problem.solve()
    return solution.value(tensions), solution.value(solver_F), solution.value(solver_T), solution.value(cost)

def rotationXnp(angle):
    RXM = numpy.eye(4)
    RXM[1, 1] = cos(angle)
    RXM[1, 2] = -sin(angle)
    RXM[2, 1] = sin(angle)
    RXM[2, 2] = cos(angle)
    return RXM
def rotationYnp(angle):
    RYM = numpy.eye(4)
    RYM[0, 0] = cos(angle)
    RYM[0, 2] = sin(angle)
    RYM[2, 0] = -sin(angle)
    RYM[2, 2] = cos(angle)
    return RYM
def rotationZnp(angle):
    RZM = numpy.eye(4)
    RZM[0, 0] = cos(angle)
    RZM[0, 1] = sin(angle)
    RZM[1, 0] = -sin(angle)
    RZM[1, 1] = cos(angle)
    return RZM

def transform_handle_poses(T, RX, RY, RZ, handle_poses):
    RXM = rotationXnp(RX)
    RYM = rotationYnp(RY)
    RZM = rotationZnp(RZ)
    R = RXM @ RYM @ RZM
    print(R)
    R[3, 0:3] = T
    transformed_poses = []
    for handle_pose in handle_poses:
        handle_pose += [1]
        # print(handle_pose)
        # print(R)
        handle_pose_mx = numpy.array([handle_pose])
        # print(handle_pose_mx)
        transformed = handle_pose_mx @ R
        print(transformed)
        transformed_poses.append(transformed[0, 0:3])
    
    return transformed_poses

def plot_3d_strings_with_transformation(tensions, T, RX, RY, RZ):
    """
    Plots the 3D representation of strings with transformation applied to handle attachment points
    and colors based on tension values.

    Args:
    tensions (list): List of tension values for each string.
    T (array): Translation vector for the end effector.
    RX (float): Rotation angle around X-axis.
    RY (float): Rotation angle around Y-axis.
    RZ (float): Rotation angle around Z-axis.
    """
    transformed_handle_poses = transform_handle_poses(T, RX, RY, RZ, handle_attachment_poses)
    
    # Normalize tension values for color mapping
    norm_tensions = (tensions - numpy.min(tensions)) / (numpy.max(tensions) - numpy.min(tensions))
    colors = plt.cm.viridis(norm_tensions)  # Use a colormap (e.g., viridis)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (base, handle) in enumerate(zip(base_attachment_poses, transformed_handle_poses)):
        print(i, base, handle)
        ax.plot(
            [base[0], handle[0]],
            [base[1], handle[1]],
            [base[2], handle[2]],
            color=colors[i]
        )
        # Add scatter points for clarity
        ax.scatter(*base, color='red', s=50, label='Base' if i == 0 else "")
        ax.scatter(*handle, color='blue', s=50, label='Handle' if i == 0 else "")
    
    norm = plt.Normalize(vmin=numpy.min(tensions), vmax=numpy.max(tensions))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Needed for colorbar to work

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Tension (N)')
               
    ax.set_title("3D Strings with Transformed Handle Poses")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()

T = [0.1, 0.3, 0.3]  # Translation vector
RX, RY, RZ = MX(0.1), MX(0.2), MX(1.3)  # Rotations in radians

cable_lengths = forward_kinematics(MX(T), RX, RY, RZ)
print("Cable Lengths:", cable_lengths)
result = inverse_kinematics(cable_lengths)
print("Solution:", result)
tensions, f, t, cost = force_kinematics(MX(T), RX, RY, RZ, MX([0, 0, 0]), MX([0, 0, 0]), 10, 1)
print("Tensions:", tensions, "\n", f, t, cost)

plot_3d_strings_with_transformation(tensions, numpy.array(T), RX, RY, RZ)