from Kinematics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import numpy


def plot_3d_video_with_imageio(initial_pose, target_pose, mass, dt, output_filename="3d_string_animation.mp4"):
    """
    Generate a video of 3D strings over time using imageio.

    Args:
    pose_array (array): A 3xN+1 array of poses [Tx, Ty, Tz] over time.
    force_array (array): A 3xN array of forces over time.
    dt (float): Time step between frames.
    output_filename (str): Output video filename.
    """
    base_poses, handle_poses = optimal6()
    # Store frames for the video
    frames = []
    frame = 0
    max_frame = 100
    current_pose = initial_pose
    current_velocity = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    error = target_pose - current_pose
    while frame < max_frame or not ((error@error.T) > (0.01)**2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.set_zlim([-1, 2])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_title(f"Frame {frame + 1} \n Error {numpy.round(error, 2)}")
        T = current_pose[0:3]
        RX = current_pose[3]
        RY = current_pose[4]
        RZ = current_pose[5]

        kP = 2*numpy.array([1, 1, 1, 0.1, 0.1, 0.1])
        kD = 3*numpy.array([1, 1, 1, 0.1, 0.1, 0.1])

        desired_forces = kP * error - kD * current_velocity
        # print("DUNGA", T, T.size)
        # RX, RY, RZ = 0, 0, 0  # Static rotations
        tensions, f, t, cost = force_kinematics(T, RX, RY, RZ, desired_forces[0:3], desired_forces[3:6], 10, 1)
        transformed_poses = transform_handle_poses(T, RX, RY, RZ, handle_poses)
        print("Frame:", frame)
        print("Error",  error@error.T)
        print("Force Error:", (desired_forces[0:3]-f))
        print("Infeasibility:", cost)
        print(current_pose, current_velocity)
        current_pose += current_velocity * dt
        current_velocity[0:3] += f/mass * dt
        current_velocity[3:6] += t/mass * dt
        error = target_pose - current_pose

        # Normalize tensions for color mapping
        norm = plt.Normalize(vmin=numpy.min(tensions), vmax=numpy.max(tensions))
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

        frame = frame + 1

    # Save the video using imageio
    imageio.mimsave(output_filename, frames, fps=int(1 / dt))


if __name__ == "__main__":
    initial_pose = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    final_pose = numpy.array([0.2, 0.1, -0.2, 0, 0, 0])
    mass = 0.2
    plot_3d_video_with_imageio(initial_pose, final_pose, mass, 0.05)