from onnxruntime import InferenceSession
import torch
import math
import numpy as np
from helperGTRS import preprocess_joint, save_obj, j2d_processing, get_bbox
from helperPoseDetector import get_2d_pose
import cv2
from renderer import Renderer
import time

image_path = 'Tests/Parshwa.jpeg'

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w/2, y + h/2, h
    # cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def render(pred_verts, pred_cam, bbox, orig_height, orig_width, orig_img, mesh_face, color):

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bbox,
        img_width=orig_width,
        img_height=orig_height
    )

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_verts,
        cam=orig_cam[0],
        color=color,
        mesh_filename=None,
        rotate=False
    )

    return renederd_img

class OptimzeCamLayer(torch.nn.Module):
    def __init__(self):
        super(OptimzeCamLayer, self).__init__()

        self.img_res = 500 / 2
        self.cam_param = torch.nn.Parameter(torch.rand((1, 3)))

    def forward(self, pose3d):
        output = pose3d[:, :, :2] + self.cam_param[None, :, 1:]
        output = output * self.cam_param[None, :, :1] * self.img_res + self.img_res
        return output

# Load Models
GTRS = InferenceSession("GTRS.onnx")
PoseDetector = torch.jit.load('PoseDetector.pt')
mesh_model_face = np.load('SMPL.npy')
joint_regressor = np.load('joint_regressor.npy')

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

video_reader = VideoReader(0)

def optimize_cam_param(pred_mesh, joint_input, bbox):
    project_net = OptimzeCamLayer()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(project_net.parameters(), lr=0.1)
    pred_3d_joint = np.matmul(joint_regressor, pred_mesh)
    project_net.train()
    target_joint, _ = j2d_processing(joint_input.copy(), (500, 500), bbox, 0, 0, None)
    target_joint = torch.Tensor(target_joint[None, :, :2])

    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(torch.Tensor(pred_3d_joint))
        # print('target_joint', target_joint[:, :17, :])
        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
    
    return project_net.cam_param[0].detach().numpy()

for img in video_reader:
    
    pose = get_2d_pose(img, PoseDetector)
    if len(pose) == 0:
        continue
    joint_input = pose[0]
    joint_img = preprocess_joint(joint_input)
    bbox = get_bbox(joint_input)
    orig_height, orig_width, _ = img.shape
    orig_img = img.copy()
    mesh = GTRS.run(None, {"joint": joint_img})[0]

    cam_param_pred = optimize_cam_param(mesh, joint_input, bbox)

    cam_param = np.ndarray((1, 3))

    mean_x = mesh[0, :, 0].min()
    mean_y = mesh[0, :, 1].min()
    mean_z = mesh[0, :, 2].min()

    cam_param[0, 1] = -1 * mean_y
    cam_param[0, 2] = mean_x
    cam_param[0, 0] = mean_z + 1

    cam_param = torch.Tensor(cam_param)

    # print(cam_param)
    # print(cam_param_pred)

    rendered_img = render(mesh[0], cam_param, bbox[None, :], orig_height, orig_width, orig_img, mesh_model_face, (0.63, 0.63, 0.87))

    cv2.imshow('Rendered Image', rendered_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_reader.cap.release()


