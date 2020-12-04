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


#############################################################################################
#                                     Ground Truth
#############################################################################################
def threeD_coordinate_map_Ma(obj):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    select_object(obj)
    # bpy.ops.object.material_slot_add()
    bpy.data.materials.new('Material_3D_Coordinate_Map_MA')
    # obj.material_slots[0].material = bpy.data.materials['Material_3D_Coordinate_Map_MA']
    obj.data.materials.append(bpy.data.materials['Material_3D_Coordinate_Map_MA'])
    mat = bpy.data.materials['Material_3D_Coordinate_Map_MA']
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    mat_node=nodes.new(type='ShaderNodeOutputMaterial')
    em_node=nodes.new(type='ShaderNodeEmission')
    geo_node=nodes.new(type='ShaderNodeNewGeometry')
    links.new(geo_node.outputs[0],em_node.inputs[0])
    links.new(em_node.outputs[0],mat_node.inputs[0])
    return    

def threeD_coordinate_map(obj,fn,num):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    select_object(obj)
    # bpy.ops.object.material_slot_add()
    bpy.data.materials.new('Material_3D_Coordinate_Map')
    obj.data.materials.append(bpy.data.materials['Material_3D_Coordinate_Map'])
    mat = bpy.data.materials['Material_3D_Coordinate_Map']
    obj.active_material = mat
    mat.use_nodes = True
    max_x, min_x, max_y, min_y, max_z, min_z = wc_coordinates(obj)
    d_dict = {}
    d_dict['max_x'] = max_x
    d_dict['min_x'] = min_x
    d_dict['max_y'] = max_y
    d_dict['min_y'] = min_y
    d_dict['max_z'] = max_z
    d_dict['min_z'] = min_z
    num = str(num)
    while len(num) < 4:
        num = '0'+ num
    with open(os.path.join(path_to_output_3dnum,fn+num+'.pkl'),'wb') as f:
        pickle.dump(d_dict,f)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    em_node=nodes.new(type='ShaderNodeEmission')
    geo_node = nodes.new(type='ShaderNodeNewGeometry')
    xyz_node = nodes.new(type="ShaderNodeSeparateXYZ")
    range_x = nodes.new(type="ShaderNodeMapRange")
    range_y = nodes.new(type="ShaderNodeMapRange")
    range_z = nodes.new(type="ShaderNodeMapRange")
    combine_rgb = nodes.new(type="ShaderNodeCombineRGB")
    length = max(max_x-min_x,max_y-min_y,max_z-min_z)
    range_x.inputs['From Min'].default_value = min_x
    range_x.inputs['From Max'].default_value = min_x + length
    range_y.inputs['From Min'].default_value = min_y
    range_y.inputs['From Max'].default_value = min_y + length
    range_z.inputs['From Min'].default_value = min_z
    range_z.inputs['From Max'].default_value = min_z + length
    range_x.inputs['To Min'].default_value = 0.1
    range_y.inputs['To Min'].default_value = 0.1
    range_z.inputs['To Min'].default_value = 0.1
    links.new(geo_node.outputs['Position'],xyz_node.inputs['Vector'])
    links.new(xyz_node.outputs['X'],range_x.inputs['Value'])
    links.new(xyz_node.outputs['Y'],range_y.inputs['Value'])
    links.new(xyz_node.outputs['Z'],range_z.inputs['Value'])
    links.new(range_x.outputs['Result'],combine_rgb.inputs['R'])
    links.new(range_y.outputs['Result'],combine_rgb.inputs['G'])
    links.new(range_z.outputs['Result'],combine_rgb.inputs['B'])
    links.new(combine_rgb.outputs['Image'],em_node.inputs[0])
    links.new(em_node.outputs[0],out_node.inputs['Surface'])
    return
############################################################################################
#                                      Pseudo Normal map
############################################################################################
def normal_map_Ma(obj):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    select_object(obj)
    #    bpy.ops.object.material_slot_add()
    bpy.data.materials.new('Material_Normal_Map_MA')
    obj.data.materials.append(bpy.data.materials['Material_Normal_Map_MA'])
    mat = bpy.data.materials['Material_Normal_Map_MA']
    #    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    mat_node=nodes.new(type='ShaderNodeOutputMaterial')
    em_node=nodes.new(type='ShaderNodeEmission')
    geo_node=nodes.new(type='ShaderNodeNewGeometry')
    links.new(geo_node.outputs[0],em_node.inputs[0])
    links.new(em_node.outputs[0],mat_node.inputs[0])
    return

def normal_map(obj):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    select_object(obj)
    #    bpy.ops.object.material_slot_add()
    bpy.data.materials.new('Material_Normal_Map')
    obj.data.materials.append(bpy.data.materials['Material_Normal_Map'])
    mat = bpy.data.materials['Material_Normal_Map']
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    mat_node=nodes.new(type='ShaderNodeOutputMaterial')
    em_node=nodes.new(type='ShaderNodeEmission')
    geo_node=nodes.new(type='ShaderNodeNewGeometry')
    xyz_node = nodes.new(type="ShaderNodeSeparateXYZ")
    math_x_node = nodes.new(type="ShaderNodeMath")
    math_x_node.operation = 'MULTIPLY_ADD'
    math_x_node.inputs[1].default_value = 0.5
    math_x_node.inputs[2].default_value = 0.5
    math_y_node = nodes.new(type="ShaderNodeMath")
    math_y_node.operation = 'MULTIPLY_ADD'
    math_y_node.inputs[1].default_value = 0.5
    math_y_node.inputs[2].default_value = 0.5
    math_z_node = nodes.new(type="ShaderNodeMath")
    math_z_node.operation = 'MULTIPLY_ADD'
    math_z_node.inputs[1].default_value = 0.5
    math_z_node.inputs[2].default_value = 0.5
    combine_rgb = nodes.new(type="ShaderNodeCombineRGB")
    links.new(geo_node.outputs['True Normal'],xyz_node.inputs['Vector'])
    links.new(xyz_node.outputs['X'],math_x_node.inputs['Value'])
    links.new(xyz_node.outputs['Y'],math_y_node.inputs['Value'])
    links.new(xyz_node.outputs['Z'],math_z_node.inputs['Value'])
    links.new(math_x_node.outputs['Value'],combine_rgb.inputs['R'])
    links.new(math_y_node.outputs['Value'],combine_rgb.inputs['G'])
    links.new(math_z_node.outputs['Value'],combine_rgb.inputs['B'])
    links.new(combine_rgb.outputs['Image'],em_node.inputs[0])
    links.new(em_node.outputs[0],mat_node.inputs['Surface'])
    return
    

def get_ground_truth_img(img_name, save_path, img_type='OPEN_EXR'):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new("CompositorNodeOutputFile")
    file_output_node.format.file_format = img_type
    file_output_node.base_path = save_path
    file_output_node.file_slots[0].path = img_name
    links.new(render_layers.outputs[0], file_output_node.inputs[0])
    return
# -----------  depth map
def depth_map(obj,fn,num):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    select_object(obj)
    bpy.data.materials.new('Material_Depth_Map')
    obj.data.materials.append(bpy.data.materials['Material_Depth_Map'])
    mat = bpy.data.materials['Material_Depth_Map']
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    mat_node=nodes.new(type='ShaderNodeOutputMaterial')
    camera_node = nodes.new(type='ShaderNodeCameraData')
    range_node = nodes.new(type="ShaderNodeMapRange")
    em_node=nodes.new(type='ShaderNodeEmission')
    camera = bpy.data.objects['Camera']
    num = str(num)
    while len(num) < 4:
        num = '0'+ num
    min_z, max_z = depth_to_camera(obj,camera)
    d_dict = {}
    d_dict['min_z'] = min_z
    d_dict['max_z'] = max_z
    with open(os.path.join(path_to_output_denum,fn+num+'.pkl'),'wb') as f:
        pickle.dump(d_dict,f)
    range_node.inputs['From Min'].default_value = min_z 
    range_node.inputs['From Max'].default_value = max_z
    range_node.inputs['To Min'].default_value = 0.1 
    links.new(camera_node.outputs['View Z Depth'],range_node.inputs['Value'])
    links.new(range_node.outputs['Result'],em_node.inputs[0])
    links.new(em_node.outputs[0],mat_node.inputs['Surface'])
    return


def get_ground_truth_img(img_name, save_path, img_type='OPEN_EXR'):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new("CompositorNodeOutputFile")
    file_output_node.format.file_format = img_type
    file_output_node.base_path = save_path
    file_output_node.file_slots[0].path = img_name
    links.new(render_layers.outputs[0], file_output_node.inputs[0])
    return