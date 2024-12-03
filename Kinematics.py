from casadi import MX, pi, sin, cos, vertcat, Opti

base_attachment_poses = [
    MX([0, 0, 0]),
    MX([1, 0, 0]),
    MX([0, 0.5, 0]),
    MX([0, 1, 1]),
    MX([1, 1, 1]),
    MX([0, 0.5, 1])
]
handle_attachment_poses = [
    MX([0.02, 0.02, 0.02]),
    MX([-0.02, 0.02, 0.02]),
    MX([0.02, 0, 0.02]),
    MX([0.02, -0.02, -0.02]),
    MX([-0.02, -0.02, -0.02]),
    MX([0.02, 0, -0.02]),
]

def forward_kinematics(T, RX, RY, RZ):
    # Rotation matrices
    RXM = MX.eye(4)
    RXM[1, 1] = cos(RX)
    RXM[1, 2] = -sin(RX)
    RXM[2, 1] = sin(RX)
    RXM[2, 2] = cos(RX)

    RYM = MX.eye(4)
    RYM[0, 0] = cos(RY)
    RYM[0, 2] = sin(RY)
    RYM[2, 0] = -sin(RY)
    RYM[2, 2] = cos(RY)

    RZM = MX.eye(4)
    RZM[0, 0] = cos(RZ)
    RZM[0, 1] = -sin(RZ)
    RZM[1, 0] = sin(RZ)
    RZM[1, 1] = cos(RZ)

    # Transformation matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T

    # Calculate cable lengths
    lengths = []
    for i in range(len(base_attachment_poses)):
        # Transform handle attachment points to world coordinates
        homogenous_handle = vertcat(handle_attachment_poses[i], 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z

        # Calculate cable length as distance between base and transformed handle point
        length = ((transformed_handle_pos - base_attachment_poses[i]).T @ 
                  (transformed_handle_pos - base_attachment_poses[i]))**0.5
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
    RXM = MX.eye(4)
    RXM[1, 1] = cos(RX)
    RXM[1, 2] = -sin(RX)
    RXM[2, 1] = sin(RX)
    RXM[2, 2] = cos(RX)

    RYM = MX.eye(4)
    RYM[0, 0] = cos(RY)
    RYM[0, 2] = sin(RY)
    RYM[2, 0] = -sin(RY)
    RYM[2, 2] = cos(RY)

    RZM = MX.eye(4)
    RZM[0, 0] = cos(RZ)
    RZM[0, 1] = -sin(RZ)
    RZM[1, 0] = sin(RZ)
    RZM[1, 1] = cos(RZ)

    # Full Transformation Matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T  # Add translation

    # Error Function
    error = 0
    for i in range(len(base_attachment_poses)):
        # Compute displacement vector
        homogenous_handle = vertcat(handle_attachment_poses[i], 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z
        displ_vec = transformed_handle_pos - base_attachment_poses[i]

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

def force_kinematics(T, RX, RY, RZ, F, tx, ty, tz):
    # Rotation matrices
    RXM = MX.eye(4)
    RXM[1, 1] = cos(RX)
    RXM[1, 2] = -sin(RX)
    RXM[2, 1] = sin(RX)
    RXM[2, 2] = cos(RX)

    RYM = MX.eye(4)
    RYM[0, 0] = cos(RY)
    RYM[0, 2] = sin(RY)
    RYM[2, 0] = -sin(RY)
    RYM[2, 2] = cos(RY)

    RZM = MX.eye(4)
    RZM[0, 0] = cos(RZ)
    RZM[0, 1] = -sin(RZ)
    RZM[1, 0] = sin(RZ)
    RZM[1, 1] = cos(RZ)

    # Transformation matrix
    TM = RXM @ RYM @ RZM
    TM[0:3, 3] = T

    # Calculate unit vectors
    unit_vec = []
    for i in range(len(base_attachment_poses)):
        # Transform handle attachment points to world coordinates
        homogenous_handle = vertcat(handle_attachment_poses[i], 1)
        transformed_handle = TM @ homogenous_handle
        transformed_handle_pos = transformed_handle[0:3]  # Extract x, y, z

        # Calculate cable length as distance between base and transformed handle point
        delta = transformed_handle_pos - base_attachment_poses[i]
        length = (delta.T @ delta)**0.5
        dir = delta/length
        unit_vec.append(dir)
    #unfinished, need to minimize error of forces torques and minimize tensions
    return unit_vec

T = MX([0.1, 0.3, 0.3])  # Translation vector
RX, RY, RZ = MX(0.1), MX(0.2), MX(0.3)  # Rotations in radians

cable_lengths = forward_kinematics(T, RX, RY, RZ)
print("Cable Lengths:", cable_lengths)
result = inverse_kinematics(cable_lengths)
print("Solution:", result)
