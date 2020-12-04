
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


########################################################################################
#                         Clear the blender environment
########################################################################################    
def reset_blend():
    """
    Reset the blender. Remove all objects, camera, lights, images, materials
    """
#    bpy.ops.wm.read_factory_settings()
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    for bpy_data_iter in (
#            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lights,
            bpy.data.images,
            bpy.data.materials
            ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data, do_unlink=True)
    return
#########################################################################################
#                          Prepare rendering settings
#########################################################################################
def prepare_scene():
    """
    Prepare the rendering scene including setting the render engine, sample ratio 
    and color management
    """
    reset_blend()
    scene=bpy.data.scenes['Scene']
    scene.render.engine='CYCLES'
    avail_devices = _cycles.available_devices('CUDA')
    print(avail_devices)
    prop = bpy.context.preferences.addons['cycles'].preferences
    prop.get_devices(prop.compute_device_type)
    prop.compute_device_type = 'CUDA'
    for device in prop.devices:
        if device.type == 'CUDA':
            print('device: ', device)
            device.use = True
    scene.cycles.samples=128
    scene.cycles.use_square_samples=False 
    scene.display_settings.display_device='sRGB'
    if random.random() > 0.4:
        scene.view_settings.view_transform='Filmic'
    else:
        scene.view_settings.view_transform='Standard'
    return

    
def prepare_rendering(frame_start=0, frame_end=250):
    """
    set gpu, render resolution, using autometic denoising, start/end frame number
    """
    bpy.ops.object.select_all(action='DESELECT')
    scene=bpy.data.scenes['Scene']
    scene.cycles.device='GPU'
    scene.view_layers['View Layer'].cycles.use_denoising=True
    scene.render.resolution_x=1080
    scene.render.resolution_y=1080
    scene.render.resolution_percentage=100
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    return

def prepare_no_env_render():
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    world=bpy.data.worlds['World']
    world.use_nodes = True
    links = world.node_tree.links
    for l in links:
        links.remove(l)
    scene=bpy.data.scenes['Scene']
    scene.cycles.use_square_samples=False
    scene.cycles.samples=128
#    scene.display_settings.display_device='None'
    scene.view_settings.view_transform='Standard'
    return

