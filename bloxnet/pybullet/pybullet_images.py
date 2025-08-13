import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

FILEPATH = os.path.dirname(os.path.abspath(__file__))
AXES_IMG_PATH = os.path.join(FILEPATH, "axes.png")


def add_isometric_axes_to_image(img):
    # open axes image with pil
    axes_img = Image.open(AXES_IMG_PATH).convert("RGBA")

    # convert img to pil
    img = Image.fromarray(img).convert("RGBA")

    # thumbnail axes image
    axes_img.thumbnail((150, 150))

    # paste axes image on img
    img.paste(axes_img, (0, 10), axes_img)

    # convert back to numpy array
    img = np.array(img)

    return img


def add_axes_to_image(
    img,
    left_axes_name="X",
    bottom_axes_name="Y",
    left_bounds=(0, 0.3),
    bottom_bounds=(-0.15, 0.15),
    tick_size=0.05,
):
    # Define the axis limits based on the image dimensions and bounds
    x_min, x_max = left_bounds
    y_min, y_max = bottom_bounds

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
    ax.axis("on")  # Ensure axes are on

    # Set aspect ratio to be equal
    ax.set_aspect("equal", "box")

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)  # Inverted y-axis to match image coordinates

    # Set ticks based on the function f(x) = tick_size * x
    ax.set_xticks(
        [
            i * tick_size
            for i in range(int(x_min / tick_size), int(x_max / tick_size) + 1)
        ]
    )
    ax.set_yticks(
        [
            i * tick_size
            for i in range(int(y_min / tick_size), int(y_max / tick_size) + 1)
        ]
    )

    # Optionally add labels
    ax.set_xlabel(bottom_axes_name)
    ax.set_ylabel(left_axes_name, rotation=0, labelpad=10)  # Horizontal y-axis label

    # Draw axis arrows
    arrowprops = dict(
        facecolor="black", edgecolor="black", arrowstyle="->", lw=1.0, mutation_scale=20
    )
    ax.annotate("", xy=(x_min, y_max), xytext=(x_min, y_min), arrowprops=arrowprops)
    ax.annotate("", xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=arrowprops)

    # Adjust padding if needed
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Draw the canvas and convert it to a NumPy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_with_axes = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image_with_axes = image_with_axes.reshape(canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return image_with_axes


def add_label(img, label, x, y, size=1.0):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size
    font_color = (255, 255, 255)  # White color
    bg_color = (0, 0, 0)  # Black color
    bg_weight = 0.5  # Background weight
    thickness = 2
    padding = 6  # Padding around text for background

    # Convert RGB to BGR (OpenCV uses BGR)
    bgr_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add text
    text = f"{label}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # adjust x and y to be in the middle of the text
    y += text_height // 2

    # Calculate text background position
    bg_x1 = x - text_width // 2 - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width // 2 + padding
    bg_y2 = y + padding

    # Draw text background
    overlay = bgr_array.copy()
    cv2.rectangle(bgr_array, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, 1 - bg_weight, bgr_array, bg_weight, 0, bgr_array)

    # Draw text
    text_x = x - text_width // 2
    text_y = y
    cv2.putText(
        bgr_array,
        text,
        (text_x, text_y),
        font,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

    return rgb_array


def add_labels(
    img, view_matrix, projection_matrix, label_visually=True, label_size=1.0
):
    # Get the number of bodies in the simulation
    num_bodies = p.getNumBodies()
    # Retrieve the body IDs
    body_ids = [p.getBodyUniqueId(i) for i in range(num_bodies)]

    # remove the id == 0 (plane)
    try:
        body_ids.remove(0)
    except:
        raise ValueError(
            "No ID == 0 found in pybullet simulation, this should be the plane"
        )

    img_width, img_height = img.shape[1], img.shape[0]

    # get segmentation mask
    _, _, _, _, mask = p.getCameraImage(
        width=img_width,
        height=img_height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )
    mask = np.array(mask, dtype=np.uint8).reshape(img_height, img_width)

    # get masks
    for id in body_ids:
        # if id not in mask, continue
        if id not in mask:
            continue

        object_mask = mask.copy()

        # set all vals equal to id to get the mask
        object_mask[mask != id] = 0

        # get the centroid of the mask
        M = cv2.moments(object_mask)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # label
        img = add_label(img, str(id), cX, cY, size=label_size)

    return img


def orthographic_projection_matrix(l, r, b, t):
    return np.array(
        [
            [2 / (r - l), 0, 0, 0],
            [0, 2 / (t - b), 0, 0],
            [0, 0, -1, 0],
            [-(r + l) / (r - l), -(t + b) / (t - b), 0, 1],
        ]
    )


def highlight_blocks(
    highlight_id,
    highlight_color=[1.0, 0.0, 0.0, 1.0],
    nonhighlight_color=[1.0, 1.0, 1.0, 1.0],
):
    # Get the number of bodies in the simulation
    num_bodies = p.getNumBodies()
    # Retrieve the body IDs
    body_ids = [p.getBodyUniqueId(i) for i in range(num_bodies)]

    for id in body_ids:
        if id != highlight_id:
            p.changeVisualShape(id, -1, rgbaColor=nonhighlight_color)
        else:
            p.changeVisualShape(id, -1, rgbaColor=highlight_color)


def get_imgs(
    labels=False,
    axes=False,
    keys=["x", "y", "z", "isometric"],
    size=0.3,
    width=640,
    height=640,
    highlight_id=None,
):
    if highlight_id is not None:
        highlight_blocks(highlight_id)

    imgs = {}

    faraway = 10  # Far away from the scene

    cam_infos = {
        "isometric": {
            "eye": [faraway, faraway, faraway + size / 2],
            "target": [0, 0, size / 2],
            "up": [0, 0, 1],
            "scale": 0.6*2.0,
            "isometric_axes": True,
            "label_size": 0.75,
        },
        "x": {
            "eye": [faraway, 0, size / 2],
            "target": [0, 0, size / 2],
            "up": [0, 0, 1],
            "scale": 0.6*1.2,
            "axes": {"left": "Z", "bottom": "Y"},
            "label_size": 1.0,
        },
        "y": {
            "eye": [0, -faraway, size / 2],
            "target": [0, 0, size / 2],
            "up": [0, 0, 1],
            "scale": 0.6*1.2,
            "axes": {"left": "Z", "bottom": "X"},
            "label_size": 1.0,
        },
        "z": {
            "eye": [0, 0, faraway],
            "target": [0, 0, size / 2],
            "up": [0, 1, 0],
            "scale": 0.6*1.2,
            "axes": {"left": "Y", "bottom": "X"},
            "label_size": 1.0,
        },
    }

    imgs = []

    for key in keys:
        if key in cam_infos:
            cam_info = cam_infos[key]
        else:
            raise ValueError(f"Camera info for key {key} not found")

        # eye = cam_info["eye"]
        # target = cam_info["target"]
        offset = np.array([0.0, 0.0, -0.07])  # negative Z moves the assembly up in-frame
        eye = (np.array(cam_info["eye"]) + offset).tolist()
        target = (np.array(cam_info["target"]) + offset).tolist()
        up = cam_info["up"]
        scale = cam_info["scale"]

        # Get the view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up
        )

        l, r = -size / 2 * scale, size / 2 * scale
        b, t = -size / 2 * scale, size / 2 * scale
        projection_matrix = orthographic_projection_matrix(l, r, b, t).reshape(16)

        _, _, rgbImg, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgbImg = np.array(rgbImg, dtype=np.uint8).reshape((height, width, 4))

        if labels:
            rgbImg = add_labels(
                rgbImg,
                view_matrix,
                projection_matrix,
                label_size=cam_info["label_size"],
            )

        if "isometric_axes" in cam_info and cam_info["isometric_axes"]:
            rgbImg = add_isometric_axes_to_image(rgbImg)

        if axes and "axes" in cam_info:
            rgbImg = add_axes_to_image(
                rgbImg,
                left_bounds=(l, r),
                bottom_bounds=(b, t),
                tick_size=0.05,
                left_axes_name=cam_info["axes"]["left"],
                bottom_axes_name=cam_info["axes"]["bottom"],
            )

        imgs.append(rgbImg)

    return imgs if len(imgs) > 1 else imgs[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bloxnet.pybullet.place_blocks_in_json import (
        create_cuboid,
        init_pybullet,
        create_cylinder,
    )
    from bloxnet.structure.structure import _create_cone, Block

    init_pybullet()

    # Reset simulation
    p.resetSimulation()
    plane = p.loadURDF("plane.urdf")  # Load plane (ground)

    # set gravity
    p.setGravity(0, 0, -9.81)

    create_cuboid("cuboid", [50, 50, 50], [0, 0, 25], 0, [0.5, 0.5, 0.5, 1])
    create_cylinder("cylinder", 25, 50, [75, 0, 25], [0.5, 0.5, 0.5, 1])
    _create_cone(
        Block(
            999,
            "cone",
            "cone",
            "cone",
            [25, 50],
            [-75, 0, 25],
            [0, 0, 0, 1],
            [0.5, 0.5, 0.5, 1],
        )
    )

    x, y, z, iso = imgs = get_imgs(labels=True, axes=True, size=0.18)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(iso)
    axs[0, 0].set_title("Isometric")
    axs[0, 1].imshow(x)
    axs[0, 1].set_title("X")
    axs[1, 0].imshow(y)
    axs[1, 0].set_title("Y")
    axs[1, 1].imshow(z)
    axs[1, 1].set_title("Z")
    plt.show()
