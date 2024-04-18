from inference import detect_2d_pose
import torch

pose, net = detect_2d_pose("Tests/Parshwa.jpeg")

script_module = torch.jit.script(net)
script_module.save("models/PoseDetector.pt")
