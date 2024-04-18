from onnxruntime import InferenceSession
import torch
import numpy as np
from helperGTRS import preprocess_joint, save_obj
from helperPoseDetector import get_2d_pose
import time

image_path = "Tests/Parshwa.jpeg"

# Load Models
GTRS = InferenceSession("GTRS.onnx")
PoseDetector = torch.jit.load("PoseDetector.pt", map_location=torch.device("cpu"))
mesh_model_face = np.load("SMPL.npy")

startPose = time.time()

pose = get_2d_pose(image_path, PoseDetector)

endPose = time.time()

joint_input = pose[0]

print("Time taken for pose detection (in milliseconds): ", (endPose - startPose) * 1000)

start = time.time()

joint_img = preprocess_joint(joint_input)
mesh = GTRS.run(None, {"joint": joint_img})[0]

end = time.time()

print("Time taken for mesh reconstruction (in milliseconds): ", (end - start) * 1000)

save_obj(mesh[0], mesh_model_face, "mesh.obj")
