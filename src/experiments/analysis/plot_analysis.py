import os
from PIL import Image
import imageio
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display


def posterior_evolution_animation():
    return

def simulation_plot_animation(simulator, theta, simulator_name, dt=0.01, T=10):

    # Create a directory for the temporary images
    os.makedirs('../../../figure_out/gif/temp_images', exist_ok=True)

    # Run the simulation
    simulation = simulator(theta, dt, T)

    # Define the number of frames for the GIF
    num_frames = 100

    # Calculate the number of simulation steps per frame
    steps_per_frame = len(simulation) // num_frames

    # Determine the fixed limits for the y-axis
    y_min = simulation.min()
    y_max = simulation.max()

    images = []
    for j in range(num_frames):
        # Get the data up to the current frame
        data = simulation[:steps_per_frame*(j+1)]

        # Create the plot
        fig, ax = plt.subplots(figsize=(2,2))
        ax.plot(np.arange(steps_per_frame*(j+1))*dt, data, label='Simulation')

        # Set the fixed limits for the x and y axes
        ax.set_xlim(0, T)
        ax.set_ylim(y_min, y_max)

        # Save the plot as a PNG image in the temporary directory
        filename = f'../../../figure_out/gif/temp_images/simulation_{j+1}.png'
        plt.savefig(filename, format="png", bbox_inches="tight", dpi=300)
        plt.close()

        images.append(filename)

    # Create the GIF
    gif_filename = f'../../../figure_out/gif/simulation_{simulator_name}.gif'
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for filename in images:
            image = Image.open(filename)
            writer.append_data(np.array(image))

    # Remove the temporary image files and directory
    for filename in images:
        os.remove(filename)
    os.rmdir('../../../figure_out/gif/temp_images')
