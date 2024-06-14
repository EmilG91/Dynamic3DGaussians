import os
import argparse
from importlib.machinery import SourceFileLoader
import torch
import pathlib
import hashlib

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814
REMOVE_BACKGROUND = False  # False or True


def rgb_to_spherical_harmonic(rgb):
    return (rgb - 0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh * C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, spherical_harmonics, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "f_rest_0",
        "f_rest_1",
        "f_rest_2",
        "f_rest_3",
        "f_rest_4",
        "f_rest_5",
        "f_rest_6",
        "f_rest_7",
        "f_rest_8",
        "f_rest_9",
        "f_rest_10",
        "f_rest_11",
        "f_rest_12",
        "f_rest_13",
        "f_rest_14",
        "f_rest_15",
        "f_rest_16",
        "f_rest_17",
        "f_rest_18",
        "f_rest_19",
        "f_rest_20",
        "f_rest_21",
        "f_rest_22",
        "f_rest_23",
        "f_rest_24",
        "f_rest_25",
        "f_rest_26",
        "f_rest_27",
        "f_rest_28",
        "f_rest_29",
        "f_rest_30",
        "f_rest_31",
        "f_rest_32",
        "f_rest_33",
        "f_rest_34",
        "f_rest_35",
        "f_rest_36",
        "f_rest_37",
        "f_rest_38",
        "f_rest_39",
        "f_rest_40",
        "f_rest_41",
        "f_rest_42",
        "f_rest_43",
        "f_rest_44",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    dtype_full = [(attribute, "f4") for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate(
        (means, normals, colors, spherical_harmonics, opacities, scales, rotations), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

    print(hashlib.md5(attributes).hexdigest())
    print(f"Saved PLY format Splat to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--path", type=pathlib.Path, help="Path to npz file")
    parser.add_argument("--no-harmonics", action="store_false", dest='harmonics', default=True, help="calculate no harmonics")
    return parser.parse_args()


def load_scene_data(path, seg_as_col=False):  # seq, exp, seg_as_col=False):
    params = dict(np.load(f"{path}"))

#    deviceType = "cuda"
#    if torch.cuda.is_available():
#        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
#    else:
#        params = {k: torch.tensor(v).float() for k, v in params.items()}
#        deviceType = "cpu"
    params = {k: torch.tensor(v).float() for k, v in params.items()}
    deviceType = "cpu"

    print(f"seg color length = {len(params['seg_colors'])}")
    is_fg = params["seg_colors"][:, 0] > 0.5
    scene_data = []
    for t in range(len(params["means3D"])):
        rendervar = {
            "means3D": params["means3D"][t],
            "colors_precomp": (
                params["rgb_colors"][t] if not seg_as_col else params["seg_colors"]
            ),
            "rotations": torch.nn.functional.normalize(params["unnorm_rotations"][t]),
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": params["log_scales"],
            "means2D": torch.zeros_like(params["means3D"][0], device=deviceType),
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


if __name__ == "__main__":
    args = parse_args()
    tmpPath = "output/pretrained/tennis/params.npz"
    if args.path is None:
        args.path = tmpPath
    scene_data, is_fg = load_scene_data(args.path)

    params = scene_data[0]

    means = params["means3D"]
    scales = params["scales"]
    rotations = params["rotations"]
    rgbs = params["colors_precomp"]
    opacities = params["opacities"]
    ply_path = os.path.join("splat.ply")

    spherical_harmonics = np.zeros([rgbs.shape[0], 45])
    if args.harmonics:
        progress = 0
        for i in range(0, rgbs.shape[0], 1):
            for j in range(0, spherical_harmonics.shape[1], 3):
                tmp = np.zeros(3)
                tmp[0] = rgbs[i, 0]
                tmp[1] = rgbs[i, 1]
                tmp[2] = rgbs[i, 2]
                tmp = rgb_to_spherical_harmonic(tmp)
                spherical_harmonics[i, j] = tmp[0]
                spherical_harmonics[i, j+1] = tmp[1]
                spherical_harmonics[i, j+2] = tmp[2]
            if i/rgbs.shape[0] > progress + 0.2:
                progress += 0.2
                print('spherical harmonics progress: ', progress)

    save_ply(ply_path, means, scales, rotations, rgbs, spherical_harmonics, opacities)

    # Load SplaTAM config
    # experiment = SourceFileLoader(
    #     os.path.basename(args.config), args.config
    # ).load_module()
    # config = experiment.config
    # work_path = config["workdir"]
    # run_name = config["run_name"]
    # params_path = os.path.join(work_path, run_name, "params.npz")
    #
    # params = dict(np.load(params_path, allow_pickle=True))
    # means = params["means3D"]
    # scales = params["log_scales"]
    # rotations = params["unnorm_rotations"]
    # rgbs = params["rgb_colors"]
    # opacities = params["logit_opacities"]
    #
    # ply_path = os.path.join(work_path, run_name, "splat.ply")
    #
    # save_ply(ply_path, means, scales, rotations, rgbs, opacities)
