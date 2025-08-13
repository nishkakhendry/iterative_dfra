import numpy as np

# Simulation environment constants
COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
    "pink":   (255/255, 105/255, 180/255, 255/255),  # Hot pink
    "orange": (255/255, 165/255,  0/255, 255/255),   # Standard orange
    "brown":  (139/255,  69/255,  19/255, 255/255),  # Saddle brown
    "gray":   (128/255, 128/255, 128/255, 255/255),  # Neutral gray
    "purple": (160/255,  32/255, 240/255, 255/255),  # Purple (violet shade)
    "white": (255/255,  255/255, 255/255, 255/255),  # White
}

PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z
