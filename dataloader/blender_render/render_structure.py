# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 06:53:01 2020

@author: WQY
"""
import bpy, _cycles
import bmesh
import random
import math
import numpy as np
from mathutils import Vector, Euler
import os
import addon_utils
import string
import pickle
from bpy_extras.object_utils import world_to_camera_view
from .tools_blender import *
from .setting_blender import *
from .obj_blender import *
from .env_blender import *
from .labels_blender import *
    
def reset_camera(obj):
    bpy.ops.object.select_all(action='DESELECT')
    camera = bpy.data.objects['Camera']
    select_object(camera)
    pos_z = random.uniform(0, 4)
    pos_x = random.uniform(-2, 2)
    pos_y = random.uniform(-2, 2)
    camera.location = (pos_x, pos_y, pos_z)
    camera.rotation_euler = (0,0,math.radians(180))
    camera.data.lens = 50 #150
    camera.constraints.new("CHILD_OF")
    camera.constraints['Child Of'].target = obj
    camera.constraints['Child Of'].subtarget = "Control"#"Group"
    select_object(camera)
    # override={'constraint':camera.constraints["Child Of"]}
    # bpy.ops.constraint.childof_set_inverse(override,constraint="Child Of",owner='OBJECT')
    camera.constraints.new("DAMPED_TRACK")
    camera.constraints['Damped Track'].target = obj
    camera.constraints['Damped Track'].subtarget = "Control"#"Group"
    camera.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
    while not to_little(obj, camera):
        camera.data.lens= camera.data.lens + 1
    while not isVisible(obj, camera):
        camera.data.lens= camera.data.lens - 1
    return

#############################################################################################
#                                Add Force
#############################################################################################
def add_force(obj,soft):
    location_x = random.uniform(-3,3)
    location_y = random.uniform(-3,3)
    location_z = random.uniform(-3,3)
    scene = bpy.data.scenes['Scene']
    scene.use_gravity = False
    if soft:
        strength = random.choice([4,13,5,6,7,9,10,12,8,15,17,20])
    else:
        strength = random.choice([2000,3000,4000,5000,6000,7000,8000,9000,1000])
    bpy.ops.object.effector_add(type='FORCE', enter_editmode=False, align='WORLD', location=(location_x, location_y, location_z))
    force = bpy.data.objects['Field']
    select_object(force)
    force.field.strength = strength * random.choice([-1,1])
    force.field.flow = 5
    force.field.noise = 5
#    force.constraints.new("TRACK_TO")
#    force.constraints['Track To'].target = obj
#    force.constraints['Track To'].up_axis = 'UP_Y'
#    force.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    force.constraints.new("DAMPED_TRACK")
    force.constraints['Damped Track'].target = obj
    force.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
    return
    

#############################################################################################
#                                   Render Pass
#############################################################################################
def render_pass(film_path='',hdr_path='', num=10, use_3d_map=True,use_glossy=True, use_normal_map=True,use_uv_map=True,use_depth_map=True,use_abedo_map=True, save_blends=True):
    fn = film_path.split('/')[-1][:-4] + '-' + \
        ''.join(random.sample(string.ascii_letters + string.digits, 6))
    label = film_path.split('/')[-1].split('.')[0]
    scene = bpy.data.scenes['Scene']
    scene.view_layers["View Layer"].use_pass_uv = True
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.view_layers["View Layer"].use_pass_mist = True
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_glossy_direct = True
    mesh = bpy.data.objects[label]
    camera = bpy.data.objects['Camera']
    min_z, max_z = depth_to_camera(mesh,camera)
    # bpy.context.scene.world.mist_settings.start = min_z
    # bpy.context.scene.world.mist_settings.depth = max_z - min_z
    num = bpy.context.scene.frame_current
    num = str(num)
    while len(num) < 4:
        num = '0'+ num
    camera_dict = {}
    ng = bpy.context.evaluated_depsgraph_get()
    cq = camera. evaluated_get(ng)
    camera_dict['location'] = list(cq.location)
    camera_dict['direction'] = list(cq.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0)))
    camera_dict['up'] = list(cq.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0)))
    camera_dict['lens'] = cq.data.lens
    with open(os.path.join(path_to_output_camera,fn+num+'.pkl'),'wb') as f:
        pickle.dump(camera_dict,f)        
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    file_output_node_img = tree.nodes.new('CompositorNodeOutputFile')
    file_output_node_img.format.file_format = 'PNG'
    file_output_node_img.base_path = path_to_output_images
    file_output_node_img.file_slots[0].path = fn
    imglk = links.new(render_layers.outputs['Image'], file_output_node_img.inputs[0])
    bpy.ops.render.render(write_still=False)
    scene=bpy.data.scenes['Scene']
    scene.render.resolution_x=448
    scene.render.resolution_y=448
    links.remove(imglk)
    if use_uv_map:
        file_output_node_uv = tree.nodes.new('CompositorNodeOutputFile')
        file_output_node_uv.format.file_format = 'OPEN_EXR'
        file_output_node_uv.base_path = path_to_output_uv
        file_output_node_uv.file_slots[0].path = fn
        uvlk = links.new(render_layers.outputs['UV'], file_output_node_uv.inputs[0])
    # if use_normal_map:
    #     file_output_node_normal = tree.nodes.new('CompositorNodeOutputFile')
    #     file_output_node_normal.format.file_format = 'OPEN_EXR'
    #     file_output_node_normal.base_path = path_to_output_normal
    #     file_output_node_normal.file_slots[0].path = fn
    #     nolk = links.new(render_layers.outputs['Normal'], file_output_node_normal.inputs[0])
    # if use_depth_map:
    #     file_output_node_depth = tree.nodes.new('CompositorNodeOutputFile')
    #     file_output_node_depth.format.file_format = 'PNG'
    #     file_output_node_depth.base_path = path_to_output_depth
    #     file_output_node_depth.file_slots[0].path = fn
    #     dplk = links.new(render_layers.outputs['Mist'], file_output_node_depth.inputs[0])#Mist
    if use_abedo_map:
        file_output_node_abedo = tree.nodes.new('CompositorNodeOutputFile')
        file_output_node_abedo.format.file_format = 'PNG'#'PNG'
        file_output_node_abedo.base_path = path_to_output_abedo
        file_output_node_abedo.file_slots[0].path = fn
        ablk = links.new(render_layers.outputs['DiffCol'], file_output_node_abedo.inputs[0])
    if use_glossy:
        file_output_node_gs = tree.nodes.new('CompositorNodeOutputFile')
        file_output_node_gs.format.file_format = 'PNG'#'PNG'
        file_output_node_gs.base_path = path_to_output_glossy
        file_output_node_gs.file_slots[0].path = fn
        gslk = links.new(render_layers.outputs['GlossDir'], file_output_node_gs.inputs[0])
    bpy.ops.render.render(write_still=False)
    #links.remove(imglk)
    links.remove(uvlk)
    # links.remove(nolk)
    links.remove(ablk)
    links.remove(gslk)
    if use_3d_map:
        prepare_no_env_render()
        obj = bpy.data.objects[label]
        threeD_coordinate_map(obj,fn,num)
        get_ground_truth_img(fn,path_to_output_3d,img_type='OPEN_EXR')
        bpy.ops.render.render(write_still=False)
    if use_depth_map:
        prepare_no_env_render()
        obj = bpy.data.objects[label]
        depth_map(obj,fn,num)
        get_ground_truth_img(fn,path_to_output_depth,img_type='OPEN_EXR')
        bpy.ops.render.render(write_still=False)
    if use_normal_map:
        prepare_no_env_render()
        bpy.data.scenes['Scene'].render.image_settings.color_depth='8'
        bpy.data.scenes['Scene'].render.image_settings.color_mode='RGB'
        bpy.data.scenes['Scene'].render.image_settings.compression=0
        obj = bpy.data.objects[label]
        normal_map(obj)
        get_ground_truth_img(fn,path_to_output_no,img_type='OPEN_EXR')
        bpy.ops.render.render(write_still=False)
    if save_blends:
        bpy.ops.wm.save_mainfile(filepath=path_to_output_blends+fn+'.blend')
        
    return fn

##################################################################################################
def render_img(film_path='',hdr_path='',use_soft = True, use_3d_map=True,use_normal_map=True,use_uv_map=True,
               use_depth_map=True,use_abedo_map=True, save_blends=True):
    prepare_scene()
    prepare_rendering()
    obj = add_film(film_path)
    if np.random.rand() < 0.5:
        print("add lighting,", end=" ")
        world_texture(hdr_path)
        add_lighting(obj, track_to=True)
    else:
        print("add back lighting,", end=" ")
        add_back_light(obj)
        add_lighting(obj, track_to=True)
    num = np.random.choice(range(10,25))
    if use_soft:
        print("use soft,", end=" ")
        add_force(obj,True)
        soft_sub_num = np.random.choice(range(8,17))
        soft_choice_num = np.random.choice(range(max(5,soft_sub_num-10),max(soft_sub_num,16)))
        v = simulation_softbody(obj, num, soft_sub_num, soft_choice_num)
    else:
        print("no soft,", end=" ")
        add_force(obj,False)
        cloth_sub_num = np.random.choice(range(12,21))
        cloth_choice_num = np.random.choice(range(5,min(cloth_sub_num,18)))
        simulation_cloth(obj, num, cloth_sub_num, cloth_choice_num)
    #fn = render_pass(film_path, hdr_path, num)
    # reset_camera(obj)
    if not v:
        return 1
    else:
        fn = render_pass(film_path, hdr_path, num)
    return
    
    
if __name__ == "__main__":
    #                       Windows
    film_dir = "/home1/quanquan/code/generate/head-texture/head-alpha/"
    hdr_dir = "/home1/quanquan/code/generate/INDOORHDR/example/"
    path_output_dir = "/home1/quanquan/code/generate/mesh_film_small_alpha/"
    path_to_output_images= path_output_dir + 'img/'
    path_to_output_uv    = path_output_dir + 'uv/'
    path_to_output_3d    = path_output_dir +  '3dmap/'
    path_to_output_normal= path_output_dir + 'compositor_normal/'
    path_to_output_depth = path_output_dir + 'depth/'
    path_to_output_blends= path_output_dir + 'blends/'
    path_to_output_abedo = path_output_dir + 'albedo/'
    path_to_output_no    = path_output_dir + 'shader_normal/'
    path_to_output_3dnum = path_output_dir + '3dmap_max_min/'
    path_to_output_denum = path_output_dir + 'depth_max_min/'
    path_to_output_camera= path_output_dir + 'camera_information/'
    path_to_output_glossy= path_output_dir + 'glossy/'
    exist_or_make(path_to_output_images) 
    exist_or_make(path_to_output_uv)
    exist_or_make(path_to_output_3d)
    exist_or_make(path_to_output_depth)
    exist_or_make(path_to_output_blends)
    exist_or_make(path_to_output_abedo)
    exist_or_make(path_to_output_no) 
    exist_or_make(path_to_output_3dnum)
    exist_or_make(path_to_output_denum)
    exist_or_make(path_to_output_camera)
    exist_or_make(path_to_output_glossy)
    addon_utils.enable('io_import_images_as_planes')
    films = []
    hdrs = []
    for film in os.listdir(film_dir):
        films.append(os.path.join(film_dir,film))
    for hdr in os.listdir(hdr_dir):
        hdrs.append(os.path.join(hdr_dir,hdr))
    film_path = np.random.choice(films)
    while film_path[-3:] != 'png':
        film_path = np.random.choice(films)        
    hdr_path = np.random.choice(hdrs)
    if np.random.rand()<0.75:
        render_img(film_path,hdr_path,True)
    else:
        render_img(film_path,hdr_path,True)
