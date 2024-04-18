from onnxruntime import InferenceSession
import numpy as np
import time
from helperGTRS import preprocess_joint
from helperPoseDetector import get_image_tensor, get_2d_pose_from_output

poseDetect = InferenceSession("PoseDetector.onnx")
GTRS = InferenceSession("GTRS.onnx")

image_path = "Tests/Samarth.jpeg"

startPose = time.time()

img_tensor, scale, pad, stride = get_image_tensor(image_path)
pose = poseDetect.run(None, {"image": np.array(img_tensor)})[0]
joint_input = get_2d_pose_from_output(pose, stride, scale, pad)

endPose = time.time()


print("Time taken for pose detection (in milliseconds): ", (endPose - startPose) * 1000)

start = time.time()

joint_img = preprocess_joint(joint_input[0])
mesh = GTRS.run(None, {"joint": joint_img})[0]

end = time.time()

print("Time taken for mesh reconstruction (in milliseconds): ", (end - start) * 1000)

print(mesh)
