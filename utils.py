import json
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pybullet
from prompting_and_structure.pipelines.whole_structure import save_pybullet_img_position_plan
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Detect if dims2 is a result of swapping two axes in dims1
def get_axis_swap(dims1, dims2):
    for i in range(3):
        for j in range(i + 1, 3):
            swapped = dims1.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            if swapped == dims2:
                return (i, j)  # rReturns the swapped indices
    return None  # No simple swap

# Extract unique dimension triplets from block name strings
def get_unique_dim_triplets(block_names):
    unique_triplets = set()
    for name in block_names:
        dims_str = name.split('-')[0]
        dims = tuple(map(int, dims_str.split('x')))
        unique_triplets.add(dims)

    result = [list(triplet) for triplet in unique_triplets]
    return result

# Get the world-space normal vector of the top face of a block
def get_top_normal(block_id):
    _, block_quat = pybullet.getBasePositionAndOrientation(block_id)
    rot_matrix = Rotation.from_quat(block_quat).as_matrix()

    # Find which block face is now pointing most upwards (dot product with [0,0,1])
    face_normals_local = [np.array([1, 0, 0]),  # +X
                          np.array([-1, 0, 0]),  # -X
                          np.array([0, 1, 0]),  # +Y
                          np.array([0, -1, 0]),  # -Y
                          np.array([0, 0, 1]),  # +Z
                          np.array([0, 0, -1])]  # -Z

    world_up = np.array([0, 0, 1])
    max_dot = -1
    top_normal_world = None
    for n_local in face_normals_local:
        n_world = rot_matrix @ n_local
        alignment = np.dot(n_world, world_up)
        if alignment > max_dot:
            max_dot = alignment
            top_normal_world = n_world
    return top_normal_world

# Compute quaternion aligning gripper -Z to a given top normal
def quat_from_top_normal(top_normal):
    """Return a quaternion that aligns gripper -Z with block top face normal."""
    z_axis = -np.array(top_normal) / np.linalg.norm(top_normal)  # gripper's -Z
    up = np.array([0, 0, 1])  # world up, used to construct orthogonal frame
    if np.allclose(z_axis, up):
        up = np.array([1, 0, 0])  # avoid degenerate case
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    return Rotation.from_matrix(rot_matrix).as_quat()

# Load JSON file from given path
def load_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Uniformly sample (x,y) in [xmin,xmax]x[ymin,ymax], but reject only the central excl_frac x excl_frac rectangle.
def sample_out_center_rect(xmin, xmax, ymin, ymax, excl_frac=0.3):
    # compute cut boundaries
    x_width = xmax - xmin
    y_width = ymax - ymin

    half_ex = excl_frac / 2.0
    cut_lo_x = xmin + (0.5 - half_ex) * x_width
    cut_hi_x = xmin + (0.5 + half_ex) * x_width
    cut_lo_y = ymin + (0.5 - half_ex) * y_width
    cut_hi_y = ymin + (0.5 + half_ex) * y_width

    while True:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        # reject if inside central box
        if not (cut_lo_x <= x <= cut_hi_x and cut_lo_y <= y <= cut_hi_y):
            return x, y
        

# Visualise sampling area and exclusion zone
def plot_sampling_area(xmin, xmax, ymin, ymax, excl_frac=0.3):
    x_width = xmax - xmin
    y_width = ymax - ymin

    half_ex = excl_frac / 2.0
    cut_lo_x = xmin + (0.5 - half_ex) * x_width
    cut_hi_x = xmin + (0.5 + half_ex) * x_width
    cut_lo_y = ymin + (0.5 - half_ex) * y_width
    cut_hi_y = ymin + (0.5 + half_ex) * y_width

    fig, ax = plt.subplots()

    # Outer allowed region
    outer_rect = patches.Rectangle((xmin, ymin), x_width, y_width, edgecolor='black', facecolor='lightblue')
    ax.add_patch(outer_rect)

    # Inner exclusion zone
    exclusion_rect = patches.Rectangle(
        (cut_lo_x, cut_lo_y), cut_hi_x - cut_lo_x, cut_hi_y - cut_lo_y,
        edgecolor='red', facecolor='white'
    )
    ax.add_patch(exclusion_rect)

    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect('equal')
    ax.set_title("Randomised Position Sampling: Allowed (Blue), Excluded (Red)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

def retake_pictures():
    structure_dir_name = input("Enter directory path: ")

    for seq_iter in range(10):
        if seq_iter != 0:
            #  parse structure_list from new assembly
            structure_list = load_from_json(f"./gpt_caching/{structure_dir_name}/replan_assembly_{seq_iter}/position_plan.json")
        else:
            # use BloxNet assembly
            structure_list = load_from_json(f"./gpt_caching/{structure_dir_name}/initial_assembly_{0}/structure.json")

        save_pybullet_img_position_plan(json_output=structure_list, structure_dir_name=structure_dir_name, seq_iter=seq_iter)

if __name__ == "__main__":
    # Run dimension swap check example
    # dims_avail = [90, 70, 20]
    # dims_this = [90,20, 70]
    # swap = get_axis_swap(dims_avail, dims_this)
    # if swap == (0, 2):
    #     print("X and Z swapped")
    # elif swap == (1, 2):
    #     print("Y and Z swapped")
    # elif swap == (0, 1):
    #     print("X and Y swapped")

    # Plot a sampling area example
    plot_sampling_area(-0.3, 0.3, -0.8, -0.2, 0.5)

    # Manually regenerate PyBullet images
    # retake_pictures()

    