from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import colorsys
import json
import argparse


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, 'GTRS', 'lib')
add_path(lib_path)

smpl_path = osp.join(this_dir, 'GTRS', 'smplpytorch')
add_path(smpl_path)

mano_path = osp.join(this_dir, 'GTRS', 'manopth')
add_path(mano_path)

import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from GTRS.demo.renderer import Renderer
from vis import vis_2d_keypoints, vis_coco_skeleton
from _mano import MANO
from smpl import SMPL

def get_model(trained_model='3dpw'):

    mesh_model = SMPL()
    joint_regressor = mesh_model.joint_regressor_coco
    joint_num = 19
    skeleton = (
        (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
        (13, 15),  # (5, 6), #(11, 12),
        (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
    flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
    graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
        build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
    model_chk_path = osp.join(osp.dirname(__file__),'GTRS/experiment/gtrs_h36m')
    if (trained_model == '3dpw'):
        model_chk_path = osp.join(osp.dirname(__file__),'GTRS/experiment/gtrs_3dpw')

    model = models.GTRS_net.get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse, mesh_model

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

def render(result, orig_height, orig_width, orig_img, mesh_face, color):
    pred_verts, pred_cam, bbox = result['mesh'], result['cam_param'][None, :], result['bbox'][None, :]

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

def optimize_cam_param(project_net, joint_input, crop_size, model, joint_regressor):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_input.copy(), (crop_size, crop_size), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).to(device)
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).to(device)

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    # estimate mesh, pose
    model.eval()
    pred_mesh, _ = model(joint_img)
    # pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_model.face.max() + 1], :]
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)

    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())
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

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['bbox'] = bbox1

    out['target'] = proj_target_joint_img

    return out

def main(args):

    input_path = args.input_path
    output_path = args.output_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse, mesh_model = get_model()
    model = model.to(device)
    joint_regressor = torch.Tensor(joint_regressor).to(device)
    coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
    project_net = models.project_net.get_model().to(device)
    joint_input = np.load(input_path)

    joint_input = joint_input.reshape(17,-1)

    input_name = input_path.split('/')[-1].split('.')[0]
    output_path = args.output_path + '/' + input_name
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Adding pelvis and neck

    lhip_idx = coco_joints_name.index('L_Hip')
    rhip_idx = coco_joints_name.index('R_Hip')
    pelvis = (joint_input[lhip_idx, :] + joint_input[rhip_idx, :]) * 0.5
    pelvis = pelvis.reshape(1, 2)
    joint_input = np.concatenate((joint_input, pelvis))

    lshoulder_idx = coco_joints_name.index('L_Shoulder')
    rshoulder_idx = coco_joints_name.index('R_Shoulder')
    neck = (joint_input[lshoulder_idx, :] + joint_input[rshoulder_idx, :]) * 0.5
    neck = neck.reshape(1, 2)
    joint_input = np.concatenate((joint_input, neck))

    if args.input_img != '.':
        orig_img = cv2.imread(args.input_img)
        orig_height, orig_width = orig_img.shape[:2]
    else:
        orig_width, orig_height = int(np.max(joint_input[:, 0]) * 1.5), int(np.max(joint_input[:, 1]) * 1.5)
        orig_img = np.zeros((orig_height, orig_width,3))

    virtual_crop_size = 500
    out = optimize_cam_param(project_net, joint_input, crop_size=virtual_crop_size, model=model, joint_regressor=joint_regressor)

    # vis mesh
    color = (0.63, 0.63, 0.87)
    rendered_img = render(out, orig_height, orig_width, orig_img, mesh_model.face, color)  # s[idx])
    cv2.imwrite(output_path + f'.png', rendered_img)
    save_obj(out['mesh'], mesh_model.face, output_path + f'.obj')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTRS Inference')
    parser.add_argument('--input_path', type=str, default='data/pose.npy', help='input pose path')
    parser.add_argument('--input_img', type=str, default='.', help='input image path')
    parser.add_argument('--output_path', type=str, default='.', help='output path')
    args = parser.parse_args()
    main(args)