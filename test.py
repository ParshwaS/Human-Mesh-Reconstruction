from inference import detect_2d_pose
import torch
import numpy as np
from helper import preprocess_joint

# Load the image
image_path = "Tests/Parshwa.jpeg"

# Detect 2D pose
pose, net = detect_2d_pose(image_path)

script_module = torch.jit.script(net)
script_module.save("models/PoseDetector.pt")

# Load Model
GTRS = torch.jit.load("models/GTRS.pt")

joint_input = pose[0]

joint_img = preprocess_joint(joint_input)

mesh, _ = GTRS(joint_img)

print(mesh)
