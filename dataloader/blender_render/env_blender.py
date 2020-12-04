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


###############################################################################################
#                               Adding Light and HDR map
###############################################################################################

def world_texture(hdr_name):
    """
    add a HDR/EXR image for the world scene
    """
    world=bpy.data.worlds['World']
    world.use_nodes = True
    links = world.node_tree.links
    nodes = world.node_tree.nodes
    for l in links:
        links.remove(l)
    for n in nodes:
        nodes.remove(n)
    world_output = nodes.new(type='ShaderNodeOutputWorld')
    background_node = nodes.new(type='ShaderNodeBackground')
    if hdr_name[-3:] == 'exr':
        background_node.inputs[1].default_value = 100
    env_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_node.image = bpy.data.images.load(hdr_name)
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.inputs[2].default_value[1] = random.uniform(0, 3.14)
    cor_node = nodes.new(type='ShaderNodeTexCoord')
    links.new(cor_node.outputs['Generated'],mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'],env_node.inputs['Vector'])
    links.new(env_node.outputs['Color'],background_node.inputs['Color'])
    links.new(background_node.outputs['Background'],world_output.inputs['Surface'])
    return


def add_lighting(obj, track_to=True):
    """
    add point/area light in scene
    """
    if np.random.rand() > 0.3:
        bpy.context.view_layer.objects.active = None
        # docrender using method
        # d = random.uniform(2, 5)
        # litpos = Vector((0, d, 0))
        # eul = Euler((0, 0, 0), 'XYZ')
        # eul.rotate_axis('Z', random.uniform(math.radians(0), math.radians(180)))
        # eul.rotate_axis('X', random.uniform(math.radians(45), math.radians(135)))
        # litpos.rotate(eul)
        # bpy.ops.object.select_all(action='DESELECT')
        # bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=litpos)
        bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(0,0,0))
        point_light = bpy.data.objects['Point']
        select_object(point_light)
        point_light.data.use_nodes = True
        pos_z = random.uniform(5, 8)
        pos_x = random.uniform(-1.5, 1.5)
        pos_y = random.uniform(-1.5, 1.5)
        point_light.location = (pos_x, pos_y, pos_z)
        nodes=point_light.data.node_tree.nodes
        links=point_light.data.node_tree.links
        for node in nodes:
            if node.type=='OUTPUT':
                output_node = node
            elif node.type=='EMISSION':
                emission_node=node
        strngth=random.uniform(1,8)
        emission_node.inputs[1].default_value=strngth
        bbody=nodes.new(type='ShaderNodeBlackbody')
        color_temp=random.uniform(2700,10200)
        bbody.inputs[0].default_value=color_temp
        links.new(bbody.outputs[0],emission_node.inputs[0])
        if track_to:
            # Track to constrain
            point_light.constraints.new("TRACK_TO")
            point_light.constraints['Track To'].target = obj#bpy.data.objects[label]
            point_light.constraints['Track To'].up_axis = 'UP_Y'
            point_light.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
            # Damped Track constrain
            # point_light.constraints.new("DAMPED_TRACK") 
            # point_light.constraints['Damped Track'].target = bpy.data.objects[label]
            # point_light.constraints['Damped Track'].subtarget = "Control"#"Group"
            # point_light.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
    else:
        # d = random.uniform(2, 4)
        # litpos = Vector((0, d, 0))
        # eul = Euler((0, 0, 0), 'XYZ')
        # eul.rotate_axis('Z', random.uniform(math.radians(0), math.radians(180)))
        # eul.rotate_axis('X', random.uniform(math.radians(45), math.radians(135)))
        # litpos.rotate(eul)
        # bpy.ops.object.light_add(type='AREA', align='WORLD', location=litpos)
        bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0,0,0))
        area_light = bpy.data.objects['Area']
        area_light.data.use_nodes = True
        pos_z = random.uniform(4, 8)
        pos_x = random.uniform(-1.5, 1.5)
        pos_y = random.uniform(-1.5, 1.5)
        area_light.location = (pos_x, pos_y, pos_z)
        area_light.data.size = random.uniform(1,3)
        nodes=area_light.data.node_tree.nodes
        links=area_light.data.node_tree.links
        for node in nodes:
            if node.type=='OUTPUT':
                output_node = node
            elif node.type=='EMISSION':
                emission_node=node
        strngth=random.uniform(1,10)
        emission_node.inputs[1].default_value=strngth
        bbody=nodes.new(type='ShaderNodeBlackbody')
        color_temp=random.uniform(4000,9500)
        bbody.inputs[0].default_value=color_temp
        links.new(bbody.outputs[0],emission_node.inputs[0])
        if track_to:
            # Track to constrain
            area_light.constraints.new("TRACK_TO")
            area_light.constraints['Track To'].target = obj#bpy.data.objects[label]
            area_light.constraints['Track To'].up_axis = 'UP_Y'
            area_light.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
            # Damped Track constrain
            # area_light.constraints.new("DAMPED_TRACK") 
            # area_light.constraints['Damped Track'].target = bpy.data.objects[label]
            # area_light.constraints['Damped Track'].subtarget = "Control"#"Group"
            # area_light.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
    return


def add_back_light(obj):
    # bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0,0,0))
    area_light = bpy.data.lights.new(name="Backlight", type='AREA')
    area_light = bpy.data.objects.new(name="Backlight", object_data=area_light)
    bpy.context.collection.objects.link(area_light)
    select_object(area_light)
    pos_z = random.uniform(-1, -4)
    area_light.location = (0,0, pos_z)
    area_light.rotation_euler = (math.radians(180),0,0)
    area_light.constraints.new("CHILD_OF")
    area_light.constraints['Child Of'].target = obj
    area_light.constraints['Child Of'].subtarget = "Control"
    area_light.data.energy = random.uniform(300,500)
    area_light.data.size = random.uniform(3,7)
    return
    
    