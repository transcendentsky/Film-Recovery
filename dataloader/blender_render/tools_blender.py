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
    cq = cam. evaluated_get(ng)
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
    cq = cam. evaluated_get(ng)
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
    cq = cam. evaluated_get(ng)
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
    cq = cam. evaluated_get(ng)
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