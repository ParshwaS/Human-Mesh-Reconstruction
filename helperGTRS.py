import numpy as np
import cv2


def process_bbox(bbox, aspect_ratio=None, scale=1.0):
    # sanitize bboxes
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + (w - 1), y + (h - 1)
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    if aspect_ratio is None:
        aspect_ratio = 384 / 288
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale  # *1.25
    bbox[3] = h * scale  # *1.25
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox


def get_bbox(joint_img):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.0
    width = xmax - xmin
    xmin = x_center - 0.5 * width  # * 1.2
    xmax = x_center + 0.5 * width  # * 1.2

    y_center = (ymin + ymax) / 2.0
    height = ymax - ymin
    ymin = y_center - 0.5 * height  # * 1.2
    ymax = y_center + 0.5 * height  # * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def j2d_processing(kp, res, bbox, rot, f, flip_pairs):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    # flip the x coordinates
    center, scale = get_center_scale(bbox)
    trans = get_affine_transform(center, scale, rot, res)

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, :2] = affine_transform(kp[i, :2].copy(), trans)

    if f:
        kp = flip_2d_joint(kp, res[0], flip_pairs)
    kp = kp.astype("float32")
    return kp, trans


def flip_2d_joint(kp, width, flip_pairs):
    kp[:, 0] = width - kp[:, 0] - 1
    """Flip keypoints."""
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    return kp


def get_center_scale(box_info):
    x, y, w, h = box_info

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

    return center, scale


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def preprocess_joint(joint_input):
    joint_input = np.delete(joint_input, (1), axis=0)
    idx = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
    joint_input = joint_input[idx]
    joint_input = joint_input.reshape(17, -1)
    lhip_idx = 11
    rhip_idx = 12
    pelvis = (joint_input[lhip_idx, :] + joint_input[rhip_idx, :]) * 0.5
    pelvis = pelvis.reshape(1, 2)
    joint_input = np.concatenate((joint_input, pelvis))
    lshoulder_idx = 5
    rshoulder_idx = 6
    neck = (joint_input[lshoulder_idx, :] + joint_input[rshoulder_idx, :]) * 0.5
    neck = neck.reshape(1, 2)
    joint_input = np.concatenate((joint_input, neck))
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(
        joint_input.copy(), (500, 500), bbox1, 0, 0, None
    )
    joint_img, _ = j2d_processing(joint_input.copy(), (384, 288), bbox2, 0, 0, None)
    joint_img = joint_img[:, :2]
    joint_img /= np.array([[384, 288]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = joint_img[None, :, :]

    return joint_img


def create_obj_string(v, f):
    obj_string = ""
    for i in range(len(v)):
        obj_string += (
            "v " + str(v[i][0]) + " " + str(v[i][1]) + " " + str(v[i][2]) + "\n"
        )
    for i in range(len(f)):
        obj_string += (
            "f "
            + str(f[i][0] + 1)
            + "/"
            + str(f[i][0] + 1)
            + " "
            + str(f[i][1] + 1)
            + "/"
            + str(f[i][1] + 1)
            + " "
            + str(f[i][2] + 1)
            + "/"
            + str(f[i][2] + 1)
            + "\n"
        )

    return obj_string


def save_obj(v, f, file_name="output.obj"):
    obj_file = open(file_name, "w")
    obj_file.write(create_obj_string(v, f))
    obj_file.close()
