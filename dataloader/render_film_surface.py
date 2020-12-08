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

#########################################################################################
#                                       TOOLS
#########################################################################################
def select_object(obj):
    """
    DEselect all current active objects and select the parameters obj
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return
    
def isVisible(mesh, cam):
    dg = bpy.context.evaluated_depsgraph_get()
    lq = mesh.evaluated_get(dg)
    ng = bpy.context.evaluated_depsgraph_get()
    cq = cam.evaluated_get(ng)
#    bpy.context.view_layer.update()     
    mat = lq.matrix_world
    #cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    # print(cam_direction)
    print(lq.data.vertices[0].co)
    for v in lq.data.vertices:
        co_ndc = world_to_camera_view(bpy.context.scene, cq, mat @ v.co)
        print(co_ndc)
        # v1 = v.co - cam_pos
        # nm_ndc = v1.angle(v.normal)
        if (co_ndc.x < 0.03 or co_ndc.x > 0.97 or co_ndc.y < 0.03 or co_ndc.y > 0.97):
            print('out of view')
            return False
    return True
def position_rotation_object(obj_name,position=(0,0,0),rotation=(0,0,0)):
    """
    set object specific position and direction
    """
    obj = bpy.data.objects[obj_name]
    select_object(obj)
    obj.location = position
    obj.rotation_euler = rotation
    return obj

def set_visible(obj):
    obj.hide_viewport = True
    obj.hide_render = True
    return

def subdivide(obj,num):
    select_object(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=num)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()
    return
def wc_coordinates(obj):
    select_object(obj)
#    bpy.ops.object.mode_set(mode="EDIT")
#    label = obj.name
#    mesh = bpy.data.meshes[label]
#    bm = bmesh.from_edit_mesh(mesh)
    dg = bpy.context.evaluated_depsgraph_get()
    lq = obj.evaluated_get(dg)
    mat = lq.matrix_world
    world_x = []
    world_y = []
    world_z = []
    for note in lq.data.vertices:
        v = mat @ note.co
        world_x.append(v[0])
        world_y.append(v[1])
        world_z.append(v[2])
    max_x = max(world_x)
    min_x = min(world_x)
    max_y = max(world_y)
    min_y = min(world_y)
    max_z = max(world_z)
    min_z = min(world_z)
    return max_x, min_x, max_y, min_y, max_z, min_z
def depth_to_camera(obj,camera):
    dg = bpy.context.evaluated_depsgraph_get()
    lq = obj.evaluated_get(dg)
    ng = bpy.context.evaluated_depsgraph_get()
    cq = camera.evaluated_get(ng)
    mat = lq.matrix_world
    min_z = 1000
    max_z = -1
    for v in lq.data.vertices:
        co_ndc = world_to_camera_view(bpy.context.scene, cq, mat @ v.co)
        max_z = max(max_z,co_ndc.z)
        min_z = min(min_z,co_ndc.z)
    return min_z, max_z 
#     doc3D-renderer   
def is_pre_Visible(mesh, cam):
    dg = bpy.context.evaluated_depsgraph_get()
    lq = mesh.evaluated_get(dg)
    ng = bpy.context.evaluated_depsgraph_get()
    cq = cam.evaluated_get(ng)
#    bpy.context.view_layer.update()
    cam_direction = cq.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))     
    mat = lq.matrix_world
    print(mat)
    print(cq.matrix_world.to_quaternion())
    #cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    # print(cam_direction)
    print(lq.data.vertices[0].co)
    ct1 = 0
    ct2 = 0
    for v in lq.data.vertices:
        nm_ndc = cam_direction.angle(mat @ v.normal)
        co_ndc = world_to_camera_view(bpy.context.scene, cq, mat @ v.co)
        print(co_ndc)
        # v1 = v.co - cam_pos
        # nm_ndc = v1.angle(v.normal)
        if (co_ndc.x < 0.03 or co_ndc.x > 0.97 or co_ndc.y < 0.03 or co_ndc.y > 0.97):
            print('out of view')
            return False
        if nm_ndc < math.radians(120):
            ct1 += 1
        if nm_ndc > math.radians(60):
            ct2 += 1
    if min(ct1, ct2) / 10000. > 0.03:
        return False      
    return True

def to_little(mesh,cam):
    dg = bpy.context.evaluated_depsgraph_get()
    lq = mesh.evaluated_get(dg)
    ng = bpy.context.evaluated_depsgraph_get()
    cq = cam.evaluated_get(ng)
#    bpy.context.view_layer.update()
    mat = lq.matrix_world
    min_x = 1.
    max_x = 0.
    min_y = 1.
    max_y = 0.
    for v in lq.data.vertices:
        co_ndc = world_to_camera_view(bpy.context.scene, cq, mat @ v.co)
        min_x = min(min_x,co_ndc.x)
        max_x = max(max_x,co_ndc.x)
        min_y = min(min_y,co_ndc.y)
        max_y = max(max_y,co_ndc.y)
    area = (max_x - min_x)*(max_y - min_y)
    if area < 0.8:
        return False
    return True
    
def select_contour(obj):
    select_vertices = []
    select_edges = [] 
    bpy.ops.object.mode_set(mode="OBJECT")
    fmesh = bmesh.new() 
    fmesh.from_mesh(obj.data) 
    fmesh.verts.ensure_lookup_table() 
    fmesh.edges.ensure_lookup_table() 
    fmesh.faces.ensure_lookup_table()
    for v in fmesh.verts:
        if len(v.link_edges) != 4:
            select_vertices.append(v)
    for e in fmesh.edges:
        if e.verts[0] in select_vertices and e.verts[1] in select_vertices: 
            select_edges.append(e.index)
    for i in select_edges:
        obj.data.edges[i].select = True
        obj.data.edges[i].crease = 1.
    fmesh.free()
    return

def too_unwarp(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    lq = obj.evaluated_get(dg)
    cam = bpy.data.objects['Camera']
    ng = bpy.context.evaluated_depsgraph_get()
    cq = cam.evaluated_get(ng)
    cam_direction = cq.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    mat = lq.matrix_world
    nos = []
    for v in lq.data.vertices:
        nos.append(mat @ v.normal)
    mean = Vector(np.mean(nos,axis=0))
    for v in nos:
        ndc = mean.angle(v)
        if ndc > math.radians(70):
            return False
    if cam_direction.angle(mean) < math.radians(110):
        return False
    return True

def exist_or_make(path):
    if not os.path.exists(path):
        os.makedirs(path)
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

    
def prepare_rendering(frame_start=0, frame_end=150):
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
    
    
    
###########################################################################################
#                                   Setting camera
###########################################################################################
#def reset_camera(label='', track_path=True,fixed_point=True):
#    """
#    set camera
#    """
#    bpy.ops.object.select_all(action='DESELECT')
##    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0))
#    camera = bpy.data.objects['Camera']
#    select_object(camera)
#    pos_z = random.uniform(2, 5)
#    pos_x = random.uniform(-0.5, 0.5)
#    pos_y = random.uniform(-0.5, 0.5)
#    camera.location = (pos_x, pos_y, pos_z)
##    pos_z = random.uniform(5, 8)
##    pos_x = random.uniform(0.5, 1.5)
##    pos_y = random.uniform(0.5, 1.5)
##    camera.location = (pos_x, pos_y, pos_z)
#    camera.data.lens = random.randint(180, 250)
#    if track_path:
#        bpy.ops.curve.primitive_nurbs_path_add(enter_editmode=False, align='WORLD', location=(pos_x, pos_y, pos_z))
##        bpy.ops.curve.primitive_bezier_circle_add(radius=2, enter_editmode=False, 
##                                                  align='WORLD', location=(pos_x, pos_y, pos_z))
#        select_object(camera)
#        camera.constraints.new("FOLLOW_PATH")
#        camera.constraints["Follow Path"].target = bpy.data.objects["NurbsPath"]#bpy.data.objects["BezierCircle"]
#        camera.constraints["Follow Path"].use_curve_follow = True
#        override={'constraint':camera.constraints["Follow Path"]}
#        bpy.ops.constraint.followpath_path_animate(override,constraint='Follow Path')
#        camera.location = (0,0,0)
#        path = bpy.data.objects["NurbsPath"]
#        path.scale[0] = 1.5
##        seed = np.random.rand()
##        if seed >0.5:
##            path.rotation_euler[0] = 1.5708
#        if fixed_point:
#            bpy.context.object.constraints["Follow Path"].use_fixed_location = True
#            bpy.context.object.constraints["Follow Path"].offset_factor = np.random.rand()
#        else:
#            bpy.context.object.constraints["Follow Path"].use_fixed_location = False
##    camera.constraints.new("TRACK_TO")
##    camera.constraints['Track To'].target = bpy.data.objects[label]
##    camera.constraints['Track To'].up_axis = 'UP_Y'
##    camera.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
#    camera.constraints.new("DAMPED_TRACK")
#    camera.constraints['Damped Track'].target = bpy.data.objects[label]
#    camera.constraints['Damped Track'].subtarget = "Group"
#    camera.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
#    camera.rotation_euler[2] = 1.5708
#    return camera

# def reset_camera(mesh):
#     bpy.ops.object.select_all(action='DESELECT')
#     camera=bpy.data.objects['Camera']

#     # sample camera config until find a valid one
#     id = 0
#     vid = False
#     # focal length
#     bpy.data.cameras['Camera'].lens = random.randint(50,60)
#     # cam position
#     d = random.uniform(3.3, 4.3)
#     campos = Vector((0, d, 0))
#     eul = Euler((0, 0, 0), 'XYZ')
#     eul.rotate_axis('Z', random.uniform(0, 3.1415 ))#3.1415
#     eul.rotate_axis('X', random.uniform(math.radians(60), math.radians(120)))
    
#     campos.rotate(eul)
#     camera.location=campos

#     while id < 50:
#         # look at pos
#         st = (d - 2.3) / 1.0 * 0.2 + 0.3
#         lookat = Vector((random.uniform(-st, st), random.uniform(-st, st), 0))
#         eul = Euler((0, 0, 0), 'XYZ')
        
#         eul.rotate_axis('X', math.atan2(lookat.y - campos.y, campos.z))
#         eul.rotate_axis('Y', math.atan2(campos.x - lookat.x, campos.z))
#         st = (d - 2.3) / 1.0 * 15 + 5.
# #        eul.rotate_axis('Z',math.radians(90))
#         eul.rotate_axis('Z', random.uniform(math.radians( - st), math.radians( + st)))
        
#         camera.rotation_euler = eul
#         bpy.context.view_layer.update()

#         if is_pre_Visible(mesh, camera):
#             vid = True
#             break
#         id += 1
#     while not to_little(mesh, camera):
#         camera.data.lens= camera.data.lens + 1
#         print(camera.data.lens)
#     while not isVisible(mesh, camera):
#         camera.data.lens= camera.data.lens - 1
#         print(camera.data.lens)
#     return vid
def reset_camera(obj):
    bpy.ops.object.select_all(action='DESELECT')
    # Note: trans
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = bpy.context.object

    # camera = bpy.data.objects['Camera']
    camera = camera_object
    select_object(camera)
    pos_z = random.uniform(2, 4)
    pos_x = random.uniform(-0.5, 0.5)
    pos_y = random.uniform(-0.5, 0.5)
    camera.location = (pos_x, pos_y, pos_z)
    camera.rotation_euler = (0,0,math.radians(180))
    camera.data.lens = 50 #150
    camera.constraints.new("CHILD_OF")
    camera.constraints['Child Of'].target = obj
    camera.constraints['Child Of'].subtarget = "Control"#"Group"
    select_object(camera)
#    override={'constraint':camera.constraints["Child Of"]}
#    bpy.ops.constraint.childof_set_inverse(override,constraint="Child Of",owner='OBJECT')
    camera.constraints.new("DAMPED_TRACK")
    camera.constraints['Damped Track'].target = obj
    camera.constraints['Damped Track'].subtarget = "Control"#"Group"
    camera.constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
    while not to_little(obj, camera):
        camera.data.lens= camera.data.lens + 1
    while not isVisible(obj, camera):
        camera.data.lens= camera.data.lens - 1
    return

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

    # Note: trans
    trans_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader = nodes.new(type='ShaderNodeMixShader')

    bsdf_node = nodes['Principled BSDF']
    bsdf_node.inputs['Metallic'].default_value=0.#random.uniform(0.6,0.7)
    bsdf_node.inputs['Roughness'].default_value=0.95
    bsdf_node.inputs['IOR'].default_value=random.uniform(1.1,1.6)
    bsdf_node.inputs['Transmission'].default_value=0.7
    bsdf_node.inputs['Specular'].default_value= 0.08
    bsdf_node.inputs['Clearcoat'].default_value=0.03#0.3
    bsdf_node.inputs['Alpha'].default_value=0.98
    #trans_node = nodes.new(type='ShaderNodeBsdfTransparent')
    #mix_node = nodes.new(type='ShaderNodeMixShader')
    #mix_node.inputs[0].default_value = 0.018
    texturecoord_node = nodes.new(type='ShaderNodeTexCoord')
    for link in links:
        links.remove(link)
    links.new(texture_node.outputs['Color'],bsdf_node.inputs['Base Color'])
    # Notes: trans
    links.new(texture_node.outputs['Alpha'], mix_shader.inputs['Fac'])
    links.new(trans_bsdf.outputs['BSDF'], mix_shader.inputs[1])
    links.new(bsdf_node.outputs['BSDF'], mix_shader.inputs[2])
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
    cq = camera.evaluated_get(ng)
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
        if not v:
            return 1
    else:
        print("no soft,", end=" ")
        add_force(obj,False)
        cloth_sub_num = np.random.choice(range(12,21))
        cloth_choice_num = np.random.choice(range(5,min(cloth_sub_num,18)))
        simulation_cloth(obj, num, cloth_sub_num, cloth_choice_num)
    #fn = render_pass(film_path, hdr_path, num)
    # reset_camera(obj)
    
    fn = render_pass(film_path, hdr_path, num)
    return
    
    
if __name__ == "__main__":
    #                       Windows
    film_dir = "/home1/quanquan/datasets/generate/head-texture/head-alpha/"
    hdr_dir = "/home1/quanquan/datasets/generate/INDOORHDR/INDOORHDR/"
    path_output_dir = "/home1/quanquan/datasets/generate/mesh_film_small_alpha_test/"
    exist_or_make(path_output_dir)
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
    #                       Linux
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
    films = np.array([x.name for x in os.scandir(film_dir) if x.name.endswith(".png")])
    hdrs  = np.array([x.name for x in os.scandir(hdr_dir)  if (x.name.endswith(".hdr") or x.name.endswith('.exr'))])

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for i in range(3000):
        film_path = film_dir + np.random.choice(films)
        hdr_path = hdr_dir + np.random.choice(hdrs)
        
        if np.random.rand()<0.5 or i==0:
            render_img(film_path,hdr_path,True)
        else:
            render_img(film_path,hdr_path,False)
