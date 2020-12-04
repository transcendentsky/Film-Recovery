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



############################################################################################
#                              Adding Film object and Shader
############################################################################################
def add_film(film_path):
#    bpy.ops.object.mode_set(mode="OBJECT")
    film_name = film_path.split('/')[-1]
    root_path = os.path.dirname(film_path)
    bpy.ops.import_image.to_plane(files=[{"name":film_name,"name":film_name}],directory=root_path,align_axis='Z+', relative=False)
    label = film_name.split('.')[0]
    film = bpy.data.objects[label]
    select_object(film)
    film.data.materials.append(bpy.data.materials[label])
    mat = bpy.data.materials[label]
    film.active_material = mat
#    pos_x = random.rand(0.5,1.5)
#    pos_y = random.random(0.5,1.5)
#    pos_z = random.random(2,5)
#    film.location = (pos_x, pos_y, pos_z)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    texture_node = nodes['Image Texture']
    texture_node.extension='EXTEND'
    output_node = nodes['Material Output']
    # Notes: modified by trans
    trans_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    
    bsdf_node = nodes['Principled BSDF']
    bsdf_node.inputs['Metallic'].default_value=0.#random.uniform(0.6,0.7)
    bsdf_node.inputs['Roughness'].default_value=0.08
    bsdf_node.inputs['IOR'].default_value=random.uniform(1.1,1.6)
    bsdf_node.inputs['Transmission'].default_value=1.0
    bsdf_node.inputs['Specular'].default_value= 0.08
    bsdf_node.inputs['Clearcoat'].default_value=0.03 #0.3
    bsdf_node.inputs['Alpha'].default_value=1.0
    #trans_node = nodes.new(type='ShaderNodeBsdfTransparent')
    #mix_node = nodes.new(type='ShaderNodeMixShader')
    #mix_node.inputs[0].default_value = 0.018
    texturecoord_node = nodes.new(type='ShaderNodeTexCoord')
    for link in links:
        links.remove(link)
    links.new(texture_node.outputs['Color'],bsdf_node.inputs['Base Color'])
    # Notes: trans
    links.new(texture_node.outputs['Alpha'], mix_shader.inputs['Fac'])
    links.new(trans_bsdf.outputs['BSDF'], mix_shader.inputs['Shader'])
    links.new(bsdf_node.outputs['BSDF'], mix_shader.inputs['Shader'])
    links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])

    # links.new(bsdf_node.outputs['BSDF'],output_node.inputs[0])
    links.new(texture_node.inputs[0],texturecoord_node.outputs[2])
    return film

#############################################################################################
#                                Physics Simulation
#############################################################################################
def simulation_softbody(obj, frame = 10, sub_num = 12, choice_num = 5):
    select_object(obj)
    subdivide(obj, sub_num)
    bpy.ops.object.mode_set(mode="EDIT")
    label = obj.name
    mesh = bpy.data.meshes[label]
    bm = bmesh.from_edit_mesh(mesh)
    choice_verts = np.random.choice(bm.verts,choice_num)
    index_list = [v.index for v in choice_verts]
    bpy.ops.object.mode_set(mode="OBJECT")
    group = obj.vertex_groups.new(name = 'Group')
    control = obj.vertex_groups.new(name = 'Control')
    if np.random.rand():
        for i in index_list:
            group.add([i], random.choice([0.75,0.85,0.8,0.9,1]), 'REPLACE')
    else:
        group.add(index_list, random.choice([0.75,0.85,0.8,0.9]), 'REPLACE')
    control.add([v.index for v in mesh.vertices], 1., 'REPLACE')
    bpy.ops.object.modifier_add(type='SOFT_BODY')
    bpy.context.object.modifiers["Softbody"].settings.vertex_group_goal = "Group"
    bpy.context.object.modifiers["Softbody"].settings.goal_spring = 0.9
    bpy.context.object.modifiers["Softbody"].settings.goal_default = 1
    bpy.context.object.modifiers["Softbody"].settings.goal_max = 1
    bpy.context.object.modifiers["Softbody"].settings.goal_min = 0.65
    bpy.context.object.modifiers["Softbody"].settings.pull = 0.9
    bpy.context.object.modifiers["Softbody"].settings.push = 0.9
    bpy.context.object.modifiers["Softbody"].settings.damping = 0
    bpy.context.object.modifiers["Softbody"].settings.bend = 10
    bpy.context.object.modifiers["Softbody"].settings.spring_length = 100
    bpy.context.object.modifiers["Softbody"].settings.use_stiff_quads = True
    bpy.context.object.modifiers["Softbody"].settings.use_self_collision = True
    bpy.ops.ptcache.bake_all(bake=True)
    bpy.context.scene.frame_set(frame)
    count = 0
    v = True
    while not too_unwarp(obj):
        count = count + 1
        frame = random.uniform(10,150)
        bpy.context.scene.frame_set(frame)
        if count == 50:
            v = False
            break
#    for i in range(1,frame+1):
#        bpy.context.scene.frame_set(i)
    select_contour(obj)
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].render_levels = 6
    bpy.context.object.modifiers["Subdivision"].quality = 6
    reset_camera(obj)
        
        
#    bpy.ops.ptcache.bake_all(bake=False)
#    reset_camera(label=obj.name)
#    group2 = obj.vertex_groups.new(name = 'Control')
#    all_index = [v.index for v in bm.verts]
#    group2.add(all_index, 1., 'REPLACE')
    return v


def simulation_cloth(obj, frame = 10, sub_num = 12, choice_num = 5):
    select_object(obj)
    subdivide(obj, sub_num)
    bpy.ops.object.mode_set(mode="EDIT")
    label = obj.name
    mesh = bpy.data.meshes[label]
    bm = bmesh.from_edit_mesh(mesh)
    choice_verts = np.random.choice(bm.verts,choice_num)
    index_list = [v.index for v in choice_verts]
    bpy.ops.object.mode_set(mode="OBJECT")
    group = obj.vertex_groups.new(name = 'Group')
    if np.random.rand()<0.6:
        for i in index_list:
            group.add([i], random.choice([0.6,0.7,0.8,0.9,1]), 'REPLACE')
    else:
        group.add(index_list, random.choice([0.6,0.7,0.8,0.9]), 'REPLACE')
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.context.object.modifiers["Cloth"].settings.quality = 15
    bpy.context.object.modifiers["Cloth"].settings.mass = 0.4
    bpy.context.object.modifiers["Cloth"].settings.tension_stiffness = 80
    bpy.context.object.modifiers["Cloth"].settings.compression_stiffness = 80
    bpy.context.object.modifiers["Cloth"].settings.shear_stiffness = 80
    bpy.context.object.modifiers["Cloth"].settings.bending_stiffness = 150
    bpy.context.object.modifiers["Cloth"].settings.tension_damping = 25
    bpy.context.object.modifiers["Cloth"].settings.compression_damping = 25
    bpy.context.object.modifiers["Cloth"].settings.shear_damping = 25
    bpy.context.object.modifiers["Cloth"].settings.air_damping = 1
    bpy.context.object.modifiers["Cloth"].settings.vertex_group_mass = "Group"
    bpy.context.scene.frame_set(frame)
    bpy.ops.ptcache.bake_all(bake=False)
    bpy.context.view_layer.update()
#    group2 = obj.vertex_groups.new(name = 'Control')
#    all_index = [v.index for v in bm.verts]
#    group2.add(all_index, 1., 'REPLACE')
    return
