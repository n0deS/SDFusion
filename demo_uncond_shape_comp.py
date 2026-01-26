# first set up which gpu to use
import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

# import libraries
import numpy as np
from termcolor import colored, cprint
# for display
from IPython.display import Image as ipy_image
from IPython.display import display

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.utils as vutils

from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif, save_mesh_as_obj

# options for the model. please check `utils/demo_util.py` for more details
from utils.demo_util import SDFusionOpt

#seed = 2023
opt = SDFusionOpt(gpu_ids=gpu_ids)
device = opt.device

# initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-snet-all.pth'
dset="snet"
opt.init_model_args(ckpt_path=ckpt_path)
opt.init_dset_args(dataset_mode=dset)
SDFusion = create_model(opt)
print(f'[*] "{SDFusion.name()}" loaded.', 'cyan')

# unconditional generation
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

ngen = 6
ddim_steps = 100
ddim_eta = 0.

sdf_gen = SDFusion.uncond(ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta)

mesh_gen = sdf_to_mesh(sdf_gen)

# vis as gif
gen_name = f'{out_dir}/uncond.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)

for name in [gen_name]:
    display(ipy_image(name))

# Shape Completion
# initialize dataset
dataroot = 'data'
dset = 'snet'
opt.init_dset_args(dataroot=dataroot, dataset_mode=dset, cat='chair')
_, test_dl, _ = CreateDataLoader(opt)
test_ds = test_dl.dataset
test_dg = get_data_generator(test_dl)

from utils.demo_util import get_partial_shape
from utils.util_3d import combine_meshes

# 987: sofa
data_ix = 0
test_data = test_ds[data_ix]

shape = test_data['sdf'].unsqueeze(0).to(device)

# specify input range. [min, max]: [-1, 1]
# default setting: given top shape.
x_min, x_max = -1, 1
y_min, y_max = 0, 1
z_min, z_max = -1, 1
xyz_dict = {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}

# visualize input and partial shape
ret = get_partial_shape(shape, xyz_dict)
shape_part, shape_missing = ret['shape_part'], ret['shape_missing']

mesh_part = sdf_to_mesh(shape_part)
mesh_missing = sdf_to_mesh(shape_missing, color=[1, .6, .6])

# print(mesh_part)
# print(mesh_missing)

mesh_comb = combine_meshes(mesh_part, mesh_missing)
# rend_mesh_comb = render_mesh(SDFusion.renderer, mesh_comb, norm=False)

# save it
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)
sc_input_name = f'{out_dir}/shape_comp_input.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_comb, nrow=3, out_name=sc_input_name)

for name in [sc_input_name]:
    display(ipy_image(name))

print('Red cuboid: missing region')

# shape completion
ngen = 6
ddim_steps = 100
ddim_eta = 0.
output_shape_comp = SDFusion.shape_comp(shape, xyz_dict, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta)

mesh_shape_comp = sdf_to_mesh(output_shape_comp)

# vis as gif
sc_output_name = f'{out_dir}/shape_comp_output.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_shape_comp, nrow=3, out_name=sc_output_name)

sc_output_obj = f'{out_dir}/shape_comp_output.obj'
save_mesh_as_obj(mesh_shape_comp, sc_output_obj)

for name in [sc_input_name, sc_output_name]:
    display(ipy_image(name))

print('[*] Red cuboid: missing region')