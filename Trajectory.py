from Kinematics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import numpy

def generateCircle(R, Z, mass, N, max_force):
    problem = Opti()
    dt = problem.variable(1)/N
    poses = problem.variable(3, N+1)
    velocities = problem.variable(3, N + 1)
    accelerations = problem.variable(3, N)
    problem.subject_to(dt > 0.001)
    for k in range(N+1):
        angle = k * 2 * numpy.pi/N
        problem.subject_to(poses[:, k] == MX([R*cos(angle) +0.5, R*sin(angle)+0.5, Z]))
    for k in range(N):
        problem.subject_to(poses[:, k+1] == poses[:, k] + dt*velocities[:, k] + dt**2/2*(accelerations[:, k]-9.81))
        problem.subject_to(velocities[:, k+1] == velocities[:, k] + dt*accelerations[:, k])
        problem.subject_to(accelerations[:, k].T @ accelerations[:, k] <= (max_force/mass)**2)
    problem.minimize(dt)
    problem.solver('ipopt')
    solution = problem.solve()
    return solution.value(poses), solution.value(accelerations) * mass, solution.value(dt)


def plot_3d_video_with_imageio(pose_array, force_array, dt, output_filename="3d_string_animation.mp4"):
    """
    Generate a video of 3D strings over time using imageio.

    Args:
    pose_array (array): A 3xN+1 array of poses [Tx, Ty, Tz] over time.
    force_array (array): A 3xN array of forces over time.
    dt (float): Time step between frames.
    output_filename (str): Output video filename.
    """
    num_frames = pose_array.shape[1]
    base_poses = [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
        [0.5, 0, 1]
    ]
    handle_poses = [
        [0.1, 0.1, 0.1],
        [-0.1, 0.1, 0.1],
        [0, -0.1, 0.1],
        [0.1, -0.1, -0.1],
        [-0.1, -0.1, -0.1],
        [0, 0.1, -0.1],
    ]

    # Store frames for the video
    frames = []

    for frame in range(num_frames-1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.set_zlim([-1, 2])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_title(f"Frame {frame + 1}/{num_frames}")
        T = pose_array[:, frame].T
        # print("DUNGA", T, T.size)
        RX, RY, RZ = 0, 0, 0  # Static rotations
        tensions, _, _, _ = force_kinematics(T, RX, RY, RZ, force_array[:, frame], MX([0, 0, 0]), 10, 1)
        transformed_poses = transform_handle_poses(T, RX, RY, RZ, handle_poses)

        # Normalize tensions for color mapping
        norm = plt.Normalize(vmin=numpy.min(force_array), vmax=numpy.max(force_array))
        colors = plt.cm.viridis(norm(tensions))

        for i, (base, handle) in enumerate(zip(base_poses, transformed_poses)):
            ax.plot(
                [base[0], handle[0]],
                [base[1], handle[1]],
                [base[2], handle[2]],
                color=colors[i]
            )
            ax.scatter(*base, color='red', s=50)
            ax.scatter(*handle, color='blue', s=50)

        # Set figure size explicitly and specify DPI
        fig.set_size_inches(6.4, 4.8)  # Default Matplotlib size
        fig.set_dpi(100)  # Explicitly set DPI

        # Render frame
        plt.tight_layout()
        fig.canvas.draw_idle()  # Ensure the canvas is updated
        plt.pause(0.001)  # Pause briefly to allow rendering (may not be strictly necessary)

        buffer = fig.canvas.buffer_rgba()

        # Calculate the correct dimensions based on DPI
        width, height = fig.canvas.get_width_height()  # Dimensions in pixels
        if len(buffer) == 0:
            raise ValueError("Buffer is empty. Ensure the figure is correctly rendered.")
        
        image = numpy.frombuffer(buffer, dtype='uint8')
        image = image.reshape((height, width, 4))  # Use actual dimensions from canvas

        # Append the frame
        frames.append(image)


    # Save the video using imageio
    imageio.mimsave(output_filename, frames, fps=int(1 / dt))


if __name__ == "__main__":
    positions, forces, dt = generateCircle(0.3, 0.5, 1, 50, 10)
    print("Generated Poses:\n", positions)
    print("Generated Forces:\n", forces)
    print("Time Step (dt):", dt)
    plot_3d_video_with_imageio(positions, forces, dt)