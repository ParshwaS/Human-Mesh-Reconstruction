import torch
import cv2
import numpy as np
from helperGTRS import preprocess_joint
from helperPoseDetector import normalize, pad_width, extract_keypoints, group_keypoints

# Load Models
GTRS = torch.jit.load('GTRS.pt')
PoseDetector = torch.jit.load('PoseDetector.pt')

# Convert to OpenVINO
GTRS.eval().to('cpu')
PoseDetector.eval().to('cpu')

#Load Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = False if device == 'cuda' else True
net = PoseDetector.eval()
if not cpu:
    net = net.cuda()
img = cv2.imread("Tests/Parshwa.jpeg", cv2.IMREAD_COLOR)
stride = 8
upsample_ratio = 4
num_keypoints = 18
net_input_height_size = 256
height, width, _ = img.shape
scale = net_input_height_size / height
pad_value=(0, 0, 0)
img_mean=np.array([128, 128, 128], np.float32)
img_scale=np.float32(1/256)
scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
scaled_img = normalize(scaled_img, img_mean, img_scale)
min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
if not cpu:
    tensor_img = tensor_img.cuda()

traced_PoseDetector = torch.onnx.export(PoseDetector, tensor_img, "PoseDetector.onnx", verbose=True, input_names = ['image'], output_names = ['stages_output'])

stages_output = PoseDetector(tensor_img)

stage2_heatmaps = stages_output[-2]
heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

stage2_pafs = stages_output[-1]
pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

total_keypoints_num = 0
all_keypoints_by_type = []
for kpt_idx in range(num_keypoints):  # 19th for bg
    total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
for kpt_id in range(all_keypoints.shape[0]):
    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
all_poses = []
for n in range(len(pose_entries)):
    if len(pose_entries[n]) == 0:
        continue
    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
    for kpt_id in range(num_keypoints):
        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
    all_poses.append(pose_keypoints)

joint_input = all_poses[0]

joint_img = preprocess_joint(joint_input)

joint_img = torch.Tensor(joint_img)

traced_GTRS = torch.onnx.export(GTRS, joint_img, "GTRS.onnx", verbose=True, input_names = ['joint'], output_names = ['mesh', '3dpose'])

mesh, _ = GTRS(joint_img)

print(mesh)