""" ======================================================================

    BlenderPy:    [PC] Rebel Galaxy Outlaw (for Blender 3.2.2)
    Author:         mariokart64n
    Date:           August 09 2022
    Version:        0.1

    ======================================================================


    ChangeLog:

    2022-08-09
        Ported from maxscript to blender
        
    2022-08-08
        Wrote it!

====================================================================== """
import bpy  # Needed to interface with blender
from bpy_extras.io_utils import ImportHelper  # needed for OT_TestOpenFilebrowser
import struct  # Needed for Binary Reader
import random
import math
from pathlib import Path  # Needed for os stuff

useOpenDialog = True


# ====================================================================================
# MAXCSRIPT FUNCTIONS
# ====================================================================================
# These function are written to mimic native functions in
# maxscript. This is to make porting my old maxscripts
# easier, so alot of these functions may be redundant..
# ====================================================================================
#

signed, unsigned = 0, 1  # Enums for read function
seek_set, seek_cur, seek_end = 0, 1, 2  # Enums for seek function
SEEK_ABS, SEEK_REL, SEEK_END = 0, 1, 2  # Enums for seek function


def deleteScene(include=[]):
    if len(include) > 0:
        # Exit and Interactions
        if bpy.context.view_layer.objects.active != None:
            bpy.ops.object.mode_set(mode='OBJECT')

        # Select All
        bpy.ops.object.select_all(action='SELECT')

        # Loop Through Each Selection
        for o in bpy.context.view_layer.objects.selected:
            for t in include:
                if o.type == t:
                    bpy.data.objects.remove(o, do_unlink=True)
                    break

        # De-Select All
        bpy.ops.object.select_all(action='DESELECT')
    return None

def rancol4():
    return (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 1.0)


def rancol3():
    return (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))

def ceil (num):
    n = float(int(num))
    if num > n: n += 1.0
    return n

def cross(vec1=(0.0, 0.0, 0.0), vec2=(0.0, 0.0, 0.0)):
    return (
        vec2[1] * vec1[2] - vec2[2] * vec1[1],
        vec2[2] * vec1[0] - vec2[0] * vec1[2],
        vec2[0] * vec1[1] - vec2[1] * vec1[0]
        )

def dot(a=(0.0, 0.0, 0.0), b=(0.0, 0.0, 0.0)):
    return sum(map(lambda pair: pair[0] * pair[1], zip(a, b)))


#def abs(val=0.0):
#    return (-val if val < 0 else val)

def sqrt(n=0.0, l=0.001):
    # x = n
    # root = 0.0
    # count = 0
    # while True:
    #    count += 1
    #    if x == 0: break
    #    root = 0.5 * (x + (n / x))
    #    if abs(root - x) < l: break
    #    x = root
    # return root
    return math.sqrt(n)

def normalize(vec=(0.0, 0.0, 0.0)):
    div = sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]))
    return (
        (vec[0] / div) if vec[0] != 0 else 0.0,
        (vec[1] / div) if vec[1] != 0 else 0.0,
        (vec[2] / div) if vec[2] != 0 else 0.0
        )

def max(val1 = 0.0, val2 = 0.0):
    return val1 if val1 > val2 else val2

def distance(vec1=(0.0, 0.0, 0.0), vec2=(0.0, 0.0, 0.0)):
    return (sqrt((pow(vec2[0] - vec1[0], 2)) + (pow(vec2[1] - vec1[1], 2)) + (pow(vec2[2] - vec1[2], 2))))


def radToDeg(radian):
    # return (radian * 57.295779513082320876798154814105170332405472466564)
    return math.degrees(radian)


def degToRad(degree):
    # return (degree * 0.017453292519943295769236907684886127134428718885417)
    return math.radians(degree)


def bit():
    def And(integer1, integer2): return (integer1 & integer2)

    def Or(integer1, integer2): return (integer1 | integer2)

    def Xor(integer1, integer2): return (integer1 ^ integer2)

    def Not(integer1): return (~integer1)

    def Get(integer1, integer2): return ((integer1 & (1 << integer2)) >> integer2)

    def Set(integer1, integer2, boolean): return (integer1 ^ ((integer1 * 0 - (int(boolean))) ^ integer1) & ((integer1 * 0 + 1) << integer2))

    def Shift(integer1, integer2): return ((integer1 >> -integer2) if integer2 < 0 else (integer1 << integer2))

    def CharAsInt(string): return ord(int(string))

    def IntAsChar(integer): return chr(int(integer))

    def IntAsHex(integer): return format(integer, 'X')

    def IntAsFloat(integer): return struct.unpack('f', integer.to_bytes(4, byteorder='little'))


def delete(objName):
    select(objName)
    bpy.ops.object.delete(use_global=False)
    
    
def delete_all():
    if( len(bpy.data.objects) != 0 ):
        bpy.ops.object.select_all(action = 'SELECT')
        bpy.ops.object.delete(use_global=False)

class dummy:
    object = None

    def __init__(self, position = (0.0, 0.0, 0.0)):
        self.object = bpy.data.objects.new("Empty", None )
        bpy.context.scene.collection.objects.link(self.object)
        self.object.empty_display_size = 1
        self.object.empty_display_type = 'CUBE'
        self.object.location = position
        
    def position(self, pos=(0.0, 0.0, 0.0)):
        if self.object != None: self.object.location = pos

    def name(self, name=""):
        if self.object != None and name != "": self.object.name = name

    def showLinks(self, enable=False):
        return enable

    def showLinksOnly(self, enable=False):
        return enable


class matrix3:
    row1 = [1.0, 0.0, 0.0]
    row2 = [0.0, 1.0, 0.0]
    row3 = [0.0, 0.0, 1.0]
    row4 = [0.0, 0.0, 0.0]

    def __init__(self, rowA=[1.0, 0.0, 0.0], rowB=[0.0, 1.0, 0.0], rowC=[0.0, 0.0, 1.0], rowD=[0.0, 0.0, 0.0]):
        if rowA == 0:
            self.row1 = [0.0, 0.0, 0.0]
            self.row2 = [0.0, 0.0, 0.0]
            self.row3 = [0.0, 0.0, 0.0]
            self.row4 = [0.0, 0.0, 0.0]
        elif rowA == 1:
            self.row1 = [1.0, 0.0, 0.0]
            self.row2 = [0.0, 1.0, 0.0]
            self.row3 = [0.0, 0.0, 1.0]
            self.row4 = [0.0, 0.0, 0.0]
        else:
            self.row1 = rowA
            self.row2 = rowB
            self.row3 = rowC
            self.row4 = rowD

    def __repr__(self):
        return (
                "matrix3([" + str(self.row1[0]) +
                ", " + str(self.row1[1]) +
                ", " + str(self.row1[2]) +
                "], [" + str(self.row2[0]) +
                ", " + str(self.row2[1]) +
                ", " + str(self.row2[2]) +
                "], [" + str(self.row3[0]) +
                ", " + str(self.row3[1]) +
                ", " + str(self.row3[2]) +
                "], [" + str(self.row4[0]) +
                ", " + str(self.row4[1]) +
                ", " + str(self.row4[2]) + "])"
        )

    def asMat3(self):
        return (
            (self.row1[0], self.row1[1], self.row1[2]),
            (self.row2[0], self.row2[1], self.row2[2]),
            (self.row3[0], self.row3[1], self.row3[2]),
            (self.row4[0], self.row4[1], self.row4[2])
        )

    def asMat4(self):
        return (
            (self.row1[0], self.row1[1], self.row1[2], 0.0),
            (self.row2[0], self.row2[1], self.row2[2], 0.0),
            (self.row3[0], self.row3[1], self.row3[2], 0.0),
            (self.row4[0], self.row4[1], self.row4[2], 1.0)
        )

    def inverse(self):
        row1_3 = 0.0
        row2_3 = 0.0
        row3_3 = 0.0
        row4_3 = 1.0
        inv = [float] * 16
        inv[0] = (self.row2[1] * self.row3[2] * row4_3 -
                  self.row2[1] * row3_3 * self.row4[2] -
                  self.row3[1] * self.row2[2] * row4_3 +
                  self.row3[1] * row2_3 * self.row4[2] +
                  self.row4[1] * self.row2[2] * row3_3 -
                  self.row4[1] * row2_3 * self.row3[2])
        inv[4] = (-self.row2[0] * self.row3[2] * row4_3 +
                  self.row2[0] * row3_3 * self.row4[2] +
                  self.row3[0] * self.row2[2] * row4_3 -
                  self.row3[0] * row2_3 * self.row4[2] -
                  self.row4[0] * self.row2[2] * row3_3 +
                  self.row4[0] * row2_3 * self.row3[2])
        inv[8] = (self.row2[0] * self.row3[1] * row4_3 -
                  self.row2[0] * row3_3 * self.row4[1] -
                  self.row3[0] * self.row2[1] * row4_3 +
                  self.row3[0] * row2_3 * self.row4[1] +
                  self.row4[0] * self.row2[1] * row3_3 -
                  self.row4[0] * row2_3 * self.row3[1])
        inv[12] = (-self.row2[0] * self.row3[1] * self.row4[2] +
                   self.row2[0] * self.row3[2] * self.row4[1] +
                   self.row3[0] * self.row2[1] * self.row4[2] -
                   self.row3[0] * self.row2[2] * self.row4[1] -
                   self.row4[0] * self.row2[1] * self.row3[2] +
                   self.row4[0] * self.row2[2] * self.row3[1])
        inv[1] = (-self.row1[1] * self.row3[2] * row4_3 +
                  self.row1[1] * row3_3 * self.row4[2] +
                  self.row3[1] * self.row1[2] * row4_3 -
                  self.row3[1] * row1_3 * self.row4[2] -
                  self.row4[1] * self.row1[2] * row3_3 +
                  self.row4[1] * row1_3 * self.row3[2])
        inv[5] = (self.row1[0] * self.row3[2] * row4_3 -
                  self.row1[0] * row3_3 * self.row4[2] -
                  self.row3[0] * self.row1[2] * row4_3 +
                  self.row3[0] * row1_3 * self.row4[2] +
                  self.row4[0] * self.row1[2] * row3_3 -
                  self.row4[0] * row1_3 * self.row3[2])
        inv[9] = (-self.row1[0] * self.row3[1] * row4_3 +
                  self.row1[0] * row3_3 * self.row4[1] +
                  self.row3[0] * self.row1[1] * row4_3 -
                  self.row3[0] * row1_3 * self.row4[1] -
                  self.row4[0] * self.row1[1] * row3_3 +
                  self.row4[0] * row1_3 * self.row3[1])
        inv[13] = (self.row1[0] * self.row3[1] * self.row4[2] -
                   self.row1[0] * self.row3[2] * self.row4[1] -
                   self.row3[0] * self.row1[1] * self.row4[2] +
                   self.row3[0] * self.row1[2] * self.row4[1] +
                   self.row4[0] * self.row1[1] * self.row3[2] -
                   self.row4[0] * self.row1[2] * self.row3[1])
        inv[2] = (self.row1[1] * self.row2[2] * row4_3 -
                  self.row1[1] * row2_3 * self.row4[2] -
                  self.row2[1] * self.row1[2] * row4_3 +
                  self.row2[1] * row1_3 * self.row4[2] +
                  self.row4[1] * self.row1[2] * row2_3 -
                  self.row4[1] * row1_3 * self.row2[2])
        inv[6] = (-self.row1[0] * self.row2[2] * row4_3 +
                  self.row1[0] * row2_3 * self.row4[2] +
                  self.row2[0] * self.row1[2] * row4_3 -
                  self.row2[0] * row1_3 * self.row4[2] -
                  self.row4[0] * self.row1[2] * row2_3 +
                  self.row4[0] * row1_3 * self.row2[2])
        inv[10] = (self.row1[0] * self.row2[1] * row4_3 -
                   self.row1[0] * row2_3 * self.row4[1] -
                   self.row2[0] * self.row1[1] * row4_3 +
                   self.row2[0] * row1_3 * self.row4[1] +
                   self.row4[0] * self.row1[1] * row2_3 -
                   self.row4[0] * row1_3 * self.row2[1])
        inv[14] = (-self.row1[0] * self.row2[1] * self.row4[2] +
                   self.row1[0] * self.row2[2] * self.row4[1] +
                   self.row2[0] * self.row1[1] * self.row4[2] -
                   self.row2[0] * self.row1[2] * self.row4[1] -
                   self.row4[0] * self.row1[1] * self.row2[2] +
                   self.row4[0] * self.row1[2] * self.row2[1])
        inv[3] = (-self.row1[1] * self.row2[2] * row3_3 +
                  self.row1[1] * row2_3 * self.row3[2] +
                  self.row2[1] * self.row1[2] * row3_3 -
                  self.row2[1] * row1_3 * self.row3[2] -
                  self.row3[1] * self.row1[2] * row2_3 +
                  self.row3[1] * row1_3 * self.row2[2])
        inv[7] = (self.row1[0] * self.row2[2] * row3_3 -
                  self.row1[0] * row2_3 * self.row3[2] -
                  self.row2[0] * self.row1[2] * row3_3 +
                  self.row2[0] * row1_3 * self.row3[2] +
                  self.row3[0] * self.row1[2] * row2_3 -
                  (self.row3[0] * row1_3 * self.row2[2]))
        inv[11] = (-self.row1[0] * self.row2[1] * row3_3 +
                   self.row1[0] * row2_3 * self.row3[1] +
                   self.row2[0] * self.row1[1] * row3_3 -
                   self.row2[0] * row1_3 * self.row3[1] -
                   self.row3[0] * self.row1[1] * row2_3 +
                   self.row3[0] * row1_3 * self.row2[1])
        inv[15] = (self.row1[0] * self.row2[1] * self.row3[2] -
                   self.row1[0] * self.row2[2] * self.row3[1] -
                   self.row2[0] * self.row1[1] * self.row3[2] +
                   self.row2[0] * self.row1[2] * self.row3[1] +
                   self.row3[0] * self.row1[1] * self.row2[2] -
                   self.row3[0] * self.row1[2] * self.row2[1])
        det = self.row1[0] * inv[0] + self.row1[1] * inv[4] + self.row1[2] * inv[8] + row1_3 * inv[12]
        if det != 0:
            det = 1.0 / det
            return (matrix3(
                [inv[0] * det, inv[1] * det, inv[2] * det],
                [inv[4] * det, inv[5] * det, inv[6] * det],
                [inv[8] * det, inv[9] * det, inv[10] * det],
                [inv[12] * det, inv[13] * det, inv[14] * det]
            ))
        else:
            return matrix3(self.row1, self.row2, self.row3, self.row4)

    def multiply(self, B):
        C = matrix3()
        A_row1_3, A_row2_3, A_row3_3, A_row4_3 = 0.0, 0.0, 0.0, 1.0
        C.row1 = [
            self.row1[0] * B.row1[0] + self.row1[1] * B.row2[0] + self.row1[2] * B.row3[0] + A_row1_3 * B.row4[0],
            self.row1[0] * B.row1[1] + self.row1[1] * B.row2[1] + self.row1[2] * B.row3[1] + A_row1_3 * B.row4[1],
            self.row1[0] * B.row1[2] + self.row1[1] * B.row2[2] + self.row1[2] * B.row3[2] + A_row1_3 * B.row4[2]
            ]
        C.row2 = [
            self.row2[0] * B.row1[0] + self.row2[1] * B.row2[0] + self.row2[2] * B.row3[0] + A_row2_3 * B.row4[0],
            self.row2[0] * B.row1[1] + self.row2[1] * B.row2[1] + self.row2[2] * B.row3[1] + A_row2_3 * B.row4[1],
            self.row2[0] * B.row1[2] + self.row2[1] * B.row2[2] + self.row2[2] * B.row3[2] + A_row2_3 * B.row4[2],
            ]
        C.row3 = [
            self.row3[0] * B.row1[0] + self.row3[1] * B.row2[0] + self.row3[2] * B.row3[0] + A_row3_3 * B.row4[0],
            self.row3[0] * B.row1[1] + self.row3[1] * B.row2[1] + self.row3[2] * B.row3[1] + A_row3_3 * B.row4[1],
            self.row3[0] * B.row1[2] + self.row3[1] * B.row2[2] + self.row3[2] * B.row3[2] + A_row3_3 * B.row4[2]
            ]
        C.row4 = [
            self.row4[0] * B.row1[0] + self.row4[1] * B.row2[0] + self.row4[2] * B.row3[0] + A_row4_3 * B.row4[0],
            self.row4[0] * B.row1[1] + self.row4[1] * B.row2[1] + self.row4[2] * B.row3[1] + A_row4_3 * B.row4[1],
            self.row4[0] * B.row1[2] + self.row4[1] * B.row2[2] + self.row4[2] * B.row3[2] + A_row4_3 * B.row4[2]
            ]
        return C

def eulerAnglesToMatrix3 (rotXangle = 0.0, rotYangle = 0.0, rotZangle = 0.0):
    # https://stackoverflow.com/a/47283530
    cosY = math.cos(rotZangle)
    sinY = math.sin(rotZangle)
    cosP = math.cos(rotYangle)
    sinP = math.sin(rotYangle)
    cosR = math.cos(rotXangle)
    sinR = math.sin(rotXangle)
    m = matrix3 (
        [cosP * cosY, cosP * sinY, -sinP],
        [sinR * cosY * sinP - sinY * cosR, cosY * cosR + sinY * sinP * sinR, cosP * sinR],
        [sinY * sinR + cosR * cosY * sinP, cosR * sinY * sinP - sinR * cosY, cosR * cosP],
        [0.0, 0.0, 0.0]
        )
    return m

class skinOps:
    mesh = None
    skin = None
    armature = None

    def __init__(self, meshObj, armObj, skinName="Skin"):
        self.mesh = meshObj
        self.armature = armObj
        if self.mesh != None:
            for m in self.mesh.modifiers:
                if m.type == "ARMATURE":
                    self.skin = m
                    break
            if self.skin == None:
                self.skin = self.mesh.modifiers.new(type="ARMATURE", name=skinName)
            self.skin.use_vertex_groups = True
            self.skin.object = self.armature
            self.mesh.parent = self.armature

    def addbone(self, boneName, update_flag=0):
        # Adds a bone to the vertex group list
        # print("boneName:\t%s" % boneName)
        vertGroup = self.mesh.vertex_groups.get(boneName)
        if not vertGroup:
            self.mesh.vertex_groups.new(name=boneName)
        return None

    def NormalizeWeights(self, weight_array, roundTo=0):
        # Makes All weights in the weight_array sum to 1.0
        # Set roundTo 0.01 to limit weight; 0.33333 -> 0.33
        n = []
        if len(weight_array) > 0:
            s = 0.0
            n = [float] * len(weight_array)
            for i in range(0, len(weight_array)):
                if roundTo != 0:
                    n[i] = (float(int(weight_array[i] * (1.0 / roundTo)))) / (1.0 / roundTo)
                else:
                    n[i] = weight_array[i]
                s += n[i]
            s = 1.0 / s
            for i in range(0, len(weight_array)):
                n[i] *= s
        return n

    def GetNumberBones(self):
        # Returns the number of bones present in the vertex group list
        num = 0
        for b in self.armature.data.bones:
            if self.mesh.vertex_groups.get(b.name):
                num += 1
        return num

    def GetNumberVertices(self):
        # Returns the number of vertices for the object the Skin modifier is applied to.
        return len(self.mesh.data.vertices)

    def ReplaceVertexWeights(self, vertex_integer, vertex_bone_array, weight_array):
        # Sets the influence of the specified bone(s) to the specified vertex.
        # Any influence weights for the bone(s) that are not specified are erased.
        # If the bones and weights are specified as arrays, the arrays must be of the same size.

        # Check that both arrays match
        numWeights = len(vertex_bone_array)
        if len(weight_array) == numWeights and numWeights > 0:

            # Erase Any Previous Weight
            for g in self.mesh.data.vertices[vertex_integer].groups:
                self.mesh.vertex_groups[g.index].add([vertex_integer], 0.0, 'REPLACE')

            # Add New Weights
            for i in range(0, numWeights):
                self.mesh.vertex_groups[vertex_bone_array[i]].add([vertex_integer], weight_array[i], 'REPLACE')
            return True
        return False

    def GetVertexWeightCount(self, vertex_integer):
        # Returns the number of bones (vertex groups) influencing the specified vertex.
        num = 0
        for g in self.mesh.vertices[vertex_integer].groups:
            # need to write more crap
            # basically i need to know if the vertex group is for a bone and is even label as deformable
            # but lzy, me fix l8tr
            num += 1
        return num

    def boneAffectLimit(self, limit):
        # Reduce the number of bone influences affecting a single vertex
        # I copied and pasted busted ass code from somewhere as an example to
        # work from... still need to write this out but personally dont have a
        # need for it
        # for v in self.mesh.vertices:

        #     # Get a list of the non-zero group weightings for the vertex
        #     nonZero = []
        #     for g in v.groups:

        #         g.weight = round(g.weight, 4)

        #         if g.weight & lt; .0001:
        #             continue

        #         nonZero.append(g)

        #     # Sort them by weight decending
        #     byWeight = sorted(nonZero, key=lambda group: group.weight)
        #     byWeight.reverse()

        #     # As long as there are more than 'maxInfluence' bones, take the lowest influence bone
        #     # and distribute the weight to the other bones.
        #     while len(byWeight) & gt; limit:

        #         #print("Distributing weight for vertex %d" % (v.index))

        #         # Pop the lowest influence off and compute how much should go to the other bones.
        #         minInfluence = byWeight.pop()
        #         distributeWeight = minInfluence.weight / len(byWeight)
        #         minInfluence.weight = 0

        #         # Add this amount to the other bones
        #         for influence in byWeight:
        #             influence.weight = influence.weight + distributeWeight

        #         # Round off the remaining values.
        #         for influence in byWeight:
        #             influence.weight = round(influence.weight, 4)
        return None

    def GetVertexWeightBoneID(self, vertex_integer, vertex_bone_integer):
        # Returns the vertex group index of the Nth bone affecting the specified vertex.

        return None

    def GetVertexWeight(self, vertex_integer, vertex_bone_integer):
        # Returns the influence of the Nth bone affecting the specified vertex.
        for v in mesh.data.vertices:  # <MeshVertex>                              https://docs.blender.org/api/current/bpy.types.MeshVertex.html
            weights = [g.weight for g in v.groups]
            boneids = [g.group for g in v.groups]
        # return [vert for vert in bpy.context.object.data.vertices if bpy.context.object.vertex_groups['vertex_group_name'].index in [i.group for i in vert.groups]]
        return [vert for vert in bpy.context.object.data.vertices if
                bpy.context.object.vertex_groups['vertex_group_name'].index in [i.group for i in vert.groups]]

    def GetVertexWeightByBoneName(self, vertex_bone_name):
        return [vert for vert in self.mesh.data.vertices if
                self.mesh.data.vertex_groups[vertex_bone_name].index in [i.group for i in vert.groups]]

    def GetSelectedBone(self):
        # Returns the index of the current selected bone in the Bone list.
        return self.mesh.vertex_groups.active_index

    def GetBoneName(self, bone_index, nameflag_index=0):
        # Returns the bone name or node name of a bone specified by ID.
        name = ""
        try:
            name = self.mesh.vertex_groups[bone_index].name
        except:
            pass
        return name

    def GetListIDByBoneID(self, BoneID_integer):
        # Returns the ListID index given the BoneID index value.
        # The VertexGroupListID index is the index into the name-sorted.
        # The BoneID index is the non-sorted index, and is the index used by other methods that require a bone index.
        index = -1
        try:
            index = self.mesh.vertex_groups[self.armature.data.bones[BoneID_integer]].index
        except:
            pass
        return index

    def GetBoneIDByListID(self, bone_index):
        # Returns the BoneID index given the ListID index value. The ListID index is the index into the name-sorted bone listbox.
        # The BoneID index is the non-sorted index, and is the index used by other methods that require a bone index
        index = -1
        try:
            index = self.armature.data.bones[self.mesh.vertex_groups[bone_index].name].index
        except:
            pass
        return index

    def weightAllVertices(self):
        # Ensure all weights have weight and that are equal to a sum of 1.0
        return None

    def clearZeroWeights(self, limit=0.0):
        # Removes weights that are a threshold
        # for v in self.mesh.vertices:
        #     nonZero = []
        #     for g in v.groups:

        #         g.weight = round(g.weight, 4)

        #         if g.weight & le; limit:
        #             continue

        #         nonZero.append(g)

        #     # Sort them by weight decending
        #     byWeight = sorted(nonZero, key=lambda group: group.weight)
        #     byWeight.reverse()

        #     # As long as there are more than 'maxInfluence' bones, take the lowest influence bone
        #     # and distribute the weight to the other bones.
        #     while len(byWeight) & gt; limit:

        #         #print("Distributing weight for vertex %d" % (v.index))

        #         # Pop the lowest influence off and compute how much should go to the other bones.
        #         minInfluence = byWeight.pop()
        #         distributeWeight = minInfluence.weight / len(byWeight)
        #         minInfluence.weight = 0

        #         # Add this amount to the other bones
        #         for influence in byWeight:
        #             influence.weight = influence.weight + distributeWeight

        #         # Round off the remaining values.
        #         for influence in byWeight:
        #             influence.weight = round(influence.weight, 4)
        return None

    def SelectBone(self, bone_integer):
        # Selects the specified bone in the Vertex Group List
        self.mesh.vertex_groups.active_index = bone_integer
        return None

    # Probably wont bother writing this unless I really need this ability
    def saveEnvelope(self):
        # Saves Weight Data to an external binary file
        return None

    def saveEnvelopeAsASCII(self):
        # Saves Weight Data to an external ASCII file
        envASCII = "ver 3\n"
        envASCII = "numberBones " + str(self.GetNumberBones()) + "\n"
        num = 0
        for b in self.armature.data.bones:
            if self.mesh.vertex_groups.get(b.name):
                envASCII += "[boneName] " + b.name + "\n"
                envASCII += "[boneID] " + str(num) + "\n"
                envASCII += "  boneFlagLock 0\n"
                envASCII += "  boneFlagAbsolute 2\n"
                envASCII += "  boneFlagSpline 0\n"
                envASCII += "  boneFlagSplineClosed 0\n"
                envASCII += "  boneFlagDrawEnveloe 0\n"
                envASCII += "  boneFlagIsOldBone 0\n"
                envASCII += "  boneFlagDead 0\n"
                envASCII += "  boneFalloff 0\n"
                envASCII += "  boneStartPoint 0.000000 0.000000 0.000000\n"
                envASCII += "  boneEndPoint 0.000000 0.000000 0.000000\n"
                envASCII += "  boneCrossSectionCount 2\n"
                envASCII += "    boneCrossSectionInner0 3.750000\n"
                envASCII += "    boneCrossSectionOuter0 13.125000\n"
                envASCII += "    boneCrossSectionU0 0.000000\n"
                envASCII += "    boneCrossSectionInner1 3.750000\n"
                envASCII += "    boneCrossSectionOuter1 13.125000\n"
                envASCII += "    boneCrossSectionU1 1.000000\n"
                num += 1
        envASCII += "[Vertex Data]\n"
        envASCII += "  nodeCount 1\n"
        envASCII += "  [baseNodeName] " + self.mesh.name + "\n"
        envASCII += "    vertexCount " + str(len(self.mesh.vertices)) + "\n"
        for v in self.mesh.vertices:
            envASCII += "    [vertex" + str(v.index) + "]\n"
            envASCII += "      vertexIsModified 0\n"
            envASCII += "      vertexIsRigid 0\n"
            envASCII += "      vertexIsRigidHandle 0\n"
            envASCII += "      vertexIsUnNormalized 0\n"
            envASCII += "      vertexLocalPosition 0.000000 0.000000 24.38106\n"
            envASCII += "      vertexWeightCount " + str(len(v.groups)) + "\n"
            envASCII += "      vertexWeight "
            for g in v.groups:
                envASCII += str(g.group) + ","
                envASCII += str(g.weight) + " "
            envASCII += "      vertexSplineData 0.000000 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000   "
        envASCII += "  numberOfExclusinList 0\n"
        return envASCII

    def loadEnvelope(self):
        # Imports Weight Data to an external Binary file
        return None

    def loadEnvelopeAsASCII(self):
        # Imports Weight Data to an external ASCII file
        return None


class boneSys:
    armature = None
    layer = None

    def __init__(self, armatureName="Skeleton", layerName="", rootName="Scene Root"):

        # Clear Any Object Selections
        # for o in bpy.context.selected_objects: o.select = False
        bpy.context.view_layer.objects.active = None

        # Get Collection (Layers)
        if self.layer == None:
            if layerName != "":
                # make collection
                self.layer = bpy.data.collections.new(layerName)
                bpy.context.scene.collection.children.link(self.layer)
            else:
                self.layer = bpy.data.collections[bpy.context.view_layer.active_layer_collection.name]

        # Check for Armature
        armName = armatureName
        if armatureName == "": armName = "Skeleton"
        self.armature = bpy.context.scene.objects.get(armName)

        if self.armature == None:
            # Create Root Bone
            root = bpy.data.armatures.new(rootName)
            root.name = rootName

            # Create Armature
            self.armature = bpy.data.objects.new(armName, root)
            self.layer.objects.link(self.armature)

        self.armature.display_type = 'WIRE'
        self.armature.show_in_front = True

    def editMode(self, enable=True):
        #
        # Data Pointers Seem to get arranged between
        # Entering and Exiting EDIT Mode, which is
        # Required to make changes to the bones
        #
        # This needs to be called beofre and after making changes
        #

        if enable:
            # Clear Any Object Selections
            bpy.context.view_layer.objects.active = None

            # Set Armature As Active Selection
            if bpy.context.view_layer.objects.active != self.armature:
                bpy.context.view_layer.objects.active = self.armature

            # Switch to Edit Mode
            if bpy.context.object.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        else:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        return None

        def count(self):
            return len(self.armature.data.bones)

    def getNodeByName(self, boneName):
        # self.editMode(True)
        node = None
        try:
            # node = self.armature.data.bones.get('boneName')
            node = self.armature.data.edit_bones[boneName]
        except:
            pass
        # self.editMode(False)
        return node

    def getChildren(self, boneName):
        childs = []
        b = self.getNodeByName(boneName)
        if b != None:
            for bone in self.armature.data.edit_bones:
                if bone.parent == b: childs.append(bone)
        return childs

    def setParent(self, boneName, parentName):
        b = self.getNodeByName(boneName)
        p = self.getNodeByName(parentName)
        if b != None and p != None:
            b.parent = p
            return True
        return False

    def getParent(self, boneName):
        par = None
        b = self.getNodeByName(boneName)
        if b != None: par = b.parent
        return par

    def getPosition(self, boneName):
        position = (0.0, 0.0, 0.0)
        b = self.getNodeByName(boneName)
        if b != None:
            position = (
                self.armature.location[0] + b.head[0],
                self.armature.location[1] + b.head[1],
                self.armature.location[2] + b.head[2],
            )
        return position

    def setPosition(self, boneName, position):
        b = self.getNodeByName(boneName)
        pos = (
            position[0] - self.armature.location[0],
            position[1] - self.armature.location[1],
            position[2] - self.armature.location[2]
        )
        if b != None and distance(b.tail, pos) > 0.0000001: b.head = pos
        return None

    def getEndPosition(self, boneName):
        position = (0.0, 0.0, 0.0)
        b = self.getNodeByName(boneName)
        if b != None:
            position = (
                self.armature.location[0] + b.tail[0],
                self.armature.location[1] + b.tail[1],
                self.armature.location[2] + b.tail[2],
            )
        return position

    def setEndPosition(self, boneName, position):
        b = self.getNodeByName(boneName)
        pos = (
            position[0] - self.armature.location[0],
            position[1] - self.armature.location[1],
            position[2] - self.armature.location[2]
        )
        if b != None and distance(b.head, pos) > 0.0000001: b.tail = pos
        return None

    def setUserProp(self, boneName, key_string, value):
        b = self.getNodeByName(boneName)
        try:
            if b != None: b[key_string] = value
            return True
        except:
            return False

    def getUserProp(self, boneName, key_string):
        value = None
        b = self.getNodeByName(boneName)
        if b != None:
            try:
                value = b[key_string]
            except:
                pass
        return value

    def setTransform(self, boneName,
                     matrix=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0))):
        b = self.getNodeByName(boneName)
        if b != None:
            b.matrix = matrix
            return True
        return False

    def setVisibility(self, boneName, visSet=(
            True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False,
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False)):
        # Assign Visible Layers
        b = self.getNodeByName(boneName)
        if b != None:
            b.layers = visSet
            return True
        return False

    def setBoneGroup(self, boneName, normalCol=(0.0, 0.0, 0.0), selctCol=(0.0, 0.0, 0.0), activeCol=(0.0, 0.0, 0.0)):
        # Create Bone Group (custom bone colours ??)
        b = self.getNodeByName(boneName)
        if b != None:
            # arm = bpy.data.objects.new("Armature", bpy.data.armatures.new("Skeleton"))
            # layer.objects.link(arm)
            # obj.parent = arm
            # bgrp = self.armature.pose.bone_groups.new(name=msh.name)
            # bgrp.color_set = 'CUSTOM'
            # bgrp.colors.normal = normalCol
            # bgrp.colors.select = selctCol
            # bgrp.colors.active = activeCol
            # for b in obj.vertex_groups.keys():
            #    self.armature.pose.bones[b].bone_group = bgrp
            return True
        return False

    def createBone(self, boneName="", startPos=(0.0, 0.0, 0.0), endPos=(0.0, 0.0, 1.0), zAxis=(1.0, 0.0, 0.0)):

        self.editMode(True)

        # Check if bone exists
        b = None
        if boneName != "":
            try:
                b = self.armature.data.edit_bones[boneName]
                return False
            except:
                pass

        if b == None:

            # Generate Bone Name
            bName = boneName
            if bName == "": bName = "Bone_" + '{:04d}'.format(len(self.armature.data.edit_bones))

            # Create Bone
            b = self.armature.data.edit_bones.new(bName)
            #b = self.armature.data.edit_bones.new(bName.decode('utf-8', 'replace'))
            b.name = bName

            # Set As Deform Bone
            b.use_deform = True

            # Set Rotation
            roll, pitch, yaw = 0.0, 0.0, 0.0
            try:
                roll = math.acos((dot(zAxis, (1, 0, 0))) / (
                        math.sqrt(((pow(zAxis[0], 2)) + (pow(zAxis[1], 2)) + (pow(zAxis[2], 2)))) * 1.0))
            except:
                pass
            try:
                pitch = math.acos((dot(zAxis, (0, 1, 0))) / (
                        math.sqrt(((pow(zAxis[0], 2)) + (pow(zAxis[1], 2)) + (pow(zAxis[2], 2)))) * 1.0))
            except:
                pass
            try:
                yaw = math.acos((dot(zAxis, (0, 0, 1))) / (
                        math.sqrt(((pow(zAxis[0], 2)) + (pow(zAxis[1], 2)) + (pow(zAxis[2], 2)))) * 1.0))
            except:
                pass

            su = math.sin(roll)
            cu = math.cos(roll)
            sv = math.sin(pitch)
            cv = math.cos(pitch)
            sw = math.sin(yaw)
            cw = math.cos(yaw)

            b.matrix = (
                (cv * cw, su * sv * cw - cu * sw, su * sw + cu * sv * cw, 0.0),
                (cv * sw, cu * cw + su * sv * sw, cu * sv * sw - su * cw, 0.0),
                (-sv, su * cv, cu * cv, 0.0),
                (startPos[0], startPos[1], startPos[2], 1.0)
            )

            # Set Length (has to be larger then 0.1?)
            b.length = 1.0
            if startPos != endPos:
                b.head = startPos
                b.tail = endPos

        # Exit Edit Mode
        self.editMode(False)
        return True
    
    def rebuildEndPositions (self, mscale=1.0):
        for b in self.armature.data.edit_bones:
            children = self.getChildren(b.name)
            if len(children) == 1:  # Only One Child, Link End to the Child
                self.setEndPosition(b.name, self.getPosition(children[0].name))
            elif len(children) > 1:  # Multiple Children, Link End to the Average Position of all Children
                childPosAvg = [0.0, 0.0, 0.0]
                for c in children:
                    childPos = self.getPosition(c.name)
                    childPosAvg[0] += childPos[0]
                    childPosAvg[1] += childPos[1]
                    childPosAvg[2] += childPos[2]
                self.setEndPosition(b.name,
                    (childPosAvg[0] / len(children),
                    childPosAvg[1] / len(children),
                    childPosAvg[2] / len(children))
                    )
            elif b.parent != None:  # No Children use inverse of parent position
                childPos = self.getPosition(b.name)
                parPos = self.getPosition(b.parent.name)
                
                boneLength = distance(parPos, childPos)
                boneLength = 0.04 * mscale
                boneNorm = normalize(
                    (childPos[0] - parPos[0],
                     childPos[1] - parPos[1],
                     childPos[2] - parPos[2])
                    )
                
                self.setEndPosition(b.name,
                     (childPos[0] + boneLength * boneNorm[0],
                      childPos[1] + boneLength * boneNorm[1],
                      childPos[2] + boneLength * boneNorm[2])
                     )
        return None


def messageBox(message="", title="Message Box", icon='INFO'):
    def draw(self, context): self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
    return None


def getNodeByName(nodeName):
    return bpy.context.scene.objects.get(nodeName)


def classof(nodeObj):
    try:
        return str(nodeObj.type)
    except:
        return None


def makeDir(folderName):
    return Path(folderName).mkdir(parents=True, exist_ok=True)


def setUserProp(node, key_string, value):
    try:
        node[key_string] = value
        return True
    except:
        return False


def getUserProp(node, key_string):
    value = None
    try:
        value = node[key_string]
    except:
        pass
    return value


def getFileSize(filename):
    return Path(filename).stat().st_size


def doesFileExist(filename):
    file = Path(filename)
    if file.is_file():
        return True
    elif file.is_dir():
        return True
    else:
        return False


def clearListener(len=64):
    for i in range(0, len): print('')


def getFiles(filepath = ""):
    files = []
    
    fpath = '.'
    pattern = "*.*"
    
    # try to split the pattern from the path
    index = filepath.rfind('/')
    if index < 0: index = filepath.rfind('\\')
    if index > -1:
        fpath = filepath[0:index + 1]
        pattern = filepath[index + 1:]
    
    #print("fpath:\t%s" % fpath)
    #print("pattern:\t%s" % pattern)
    
    currentDirectory = Path(fpath)
    for currentFile in currentDirectory.glob(pattern):
        files.append(currentFile)


    return files


def filenameFromPath(file):  # returns: "myImage.jpg"
    return Path(file).name


def getFilenamePath(file):  # returns: "g:\subdir1\subdir2\"
    return (str(Path(file).resolve().parent) + "\\")


def getFilenameFile(file):  # returns: "myImage"
    return Path(file).stem


def getFilenameType(file):  # returns: ".jpg"
    return Path(file).suffix


def toUpper(string):
    return string.upper()


def toLower(string):
    return string.upper()

def padString(string, length=2, padChar="0", toLeft=True):
    s = str(string)
    if len(s) > length:
        s = s[0:length]
    else:
        p = ""
        for i in range(0, length): p += padChar
        if toLeft:
            s = p + s
            s = s[len(s) - length: length + 1]
        else:
            s = s + p
            s = s[0: length]
    return s


def filterString(string, string_search):
    for s in enumerate(string_search):
        string.replace(s[1], string_search[0])
    return string.split(string_search[0])


def findString(string="", token_string=""):
    return string.find(token_string)


def findItem(array, value):
    index = -1
    try:
        index = array.index(value)
    except:
        pass
    return index


def append(array, value):
    array.append(value)
    return None


def appendIfUnique(array, value):
    try:
        array.index(value)
    except:
        array.append(value)
    return None


class StandardMaterial:
    data = None
    bsdf = None
    
    maxWidth = 2048
    nodeHeight = 512
    nodeWidth = 256
    nodePos = [-256, 256.0]
    
    def __init__(self, name="Material"):
        # make material
        self.nodePos[0] -= self.nodeWidth
        self.data = bpy.data.materials.new(name=name)
        self.data.use_nodes = True
        self.data.use_backface_culling = True
        self.bsdf = self.data.node_tree.nodes["Principled BSDF"]
        self.bsdf.label = "Standard"
        self.nodePos = [-256, 256.0]
        return None

    def addNodeArea(self, nodeObj):
        nodeObj.location.x = self.nodePos[0]
        nodeObj.location.y = self.nodePos[1]
        self.nodePos[0] -= self.nodeWidth
        
        if nodeObj.dimensions[1] > self.nodeHeight:
            self.nodeHeight = nodeObj.dimensions[1]
        
        if abs(nodeObj.location.x) > self.maxWidth:
            self.nodePos[0] = -256
            self.nodePos[1] -= self.nodeHeight
            self.nodeHeight = 512

    def add(self, node_type):
        nodeObj = self.data.node_tree.nodes.new(node_type)
        self.addNodeArea(nodeObj)
        return nodeObj
    
    def attach(self, node_out, node_in):
        self.data.node_tree.links.new(node_in, node_out)
        return None
    
    def detach(self, node_con):
        #self.data.node_tree.links.remove(node_con.links[0])
        return None

    def AddColor(self, name="", colour=(0.0, 0.0, 0.0, 0.0)):
        rgbaColor = self.data.node_tree.nodes.new('ShaderNodeRGB')
        self.addNodeArea(rgbaColor)
        if name !="":
            rgbaColor.label = name
        rgbaColor.outputs[0].default_value = (colour[0], colour[1], colour[2], colour[3])
        if self.bsdf != None and self.bsdf.inputs['Base Color'] == None:
            self.data.node_tree.links.new(self.bsdf.inputs['Base Color'], rgbaColor.outputs['Color'])
        return rgbaColor
    
    def Bitmaptexture(self, filename="", alpha=False, name="ShaderNodeTexImage"):
        imageTex = self.data.node_tree.nodes.new('ShaderNodeTexImage')
        imageTex.label = name
        self.addNodeArea(imageTex)
        try:
            imageTex.image = bpy.data.images.load(
                filepath=filename,
                check_existing=False
                )
            imageTex.image.name = filenameFromPath(filename)
            imageTex.image.colorspace_settings.name = 'sRGB'
            if not alpha:
                imageTex.image.alpha_mode = 'NONE'
            else:
                imageTex.image.alpha_mode = 'STRAIGHT' # PREMUL
        except:
            imageTex.image = bpy.data.images.new(
                name=filename,
                width=8,
                height=8,
                alpha=False,
                float_buffer=False
                )
        return imageTex

    def diffuseMap(self, imageTex=None, alpha=False, name="ShaderNodeTexImage"):
        imageMap = None
        if imageTex != None and self.bsdf != None:
            imageMap = self.Bitmaptexture(filename=imageTex, alpha=alpha, name=name)
            self.data.node_tree.links.new(self.bsdf.inputs['Base Color'], imageMap.outputs['Color'])
        return imageMap

    def opacityMap(self, imageTex=None, name="ShaderNodeTexImage"):
        imageMap = None
        if imageTex != None and self.bsdf != None:
            self.data.blend_method = 'BLEND'
            self.data.shadow_method = 'HASHED'
            self.data.show_transparent_back = False
            imageMap = self.Bitmaptexture(filename=imageTex, alpha=True, name=name)
            self.data.node_tree.links.new(self.bsdf.inputs['Alpha'], imageMap.outputs['Alpha'])
        return imageMap

    def normalMap(self, imageTex=None, alpha=False, name="ShaderNodeTexImage"):
        imageMap = None
        if imageTex != None and self.bsdf != None:
            imageMap = self.Bitmaptexture(filename=imageTex, alpha=alpha, name=name)
            imageMap.image.colorspace_settings.name = 'Linear'
            normMap = self.add('ShaderNodeNormalMap')
            normMap.label = 'ShaderNodeNormalMap'
            self.attach(imageMap.outputs['Color'], normMap.inputs['Color'])
            self.attach(normMap.outputs['Normal'], self.bsdf.inputs['Normal'])
        return imageMap

    def specularMap(self, imageNode=None, invert=True, alpha=False, name="ShaderNodeTexImage"):
        imageMap = None
        if imageTex != None and self.bsdf != None:
            imageMap = self.Bitmaptexture(filename=imageTex, alpha=True, name=name)
            if invert:
                invertRGB = self.add('ShaderNodeInvert')
                invertRGB.label = 'ShaderNodeInvert'
                self.data.node_tree.links.new(invertRGB.inputs['Color'], imageMap.outputs['Color'])
                self.data.node_tree.links.new(self.bsdf.inputs['Roughness'], invertRGB.outputs['Color'])
            else:
                self.data.node_tree.links.new(self.bsdf.inputs['Roughness'], imageMap.outputs['Color'])
        return imageMap
        
    def pack_nodes_partition(self, array, begin, end):
        pivot = begin
        for i in range(begin+1, end+1):
            if array[i].dimensions[1] >= array[begin].dimensions[1]:
                pivot += 1
                array[i], array[pivot] = array[pivot], array[i]
        array[pivot], array[begin] = array[begin], array[pivot]
        return pivot

    def pack_nodes_qsort(self, array, begin=0, end=None):
        if end is None:
            end = len(array) - 1
        def _quicksort(array, begin, end):
            if begin >= end:
                return
            pivot = self.pack_nodes_partition(array, begin, end)
            _quicksort(array, begin, pivot-1)
            _quicksort(array, pivot+1, end)
        return _quicksort(array, begin, end)

    def pack_nodes (self, boxes = [], areaRatio = 0.95, padding = 0.0):
        # https://observablehq.com/@mourner/simple-rectangle-packing
        bArea = 0
        maxWidth = 0
        for i in range(0, len(boxes)):
            bArea += (boxes[i].dimensions.x + padding) * (boxes[i].dimensions.y + padding)
            maxWidth = max(maxWidth, (boxes[i].dimensions.x + padding))

        self.pack_nodes_qsort(boxes) 
        startWidth = max(ceil(sqrt(bArea / areaRatio)), maxWidth)
        spaces = [[0, 0, 0, startWidth, startWidth * 2]]
        last = []
        for i in range(0, len(boxes)):
            for p in range(len(spaces) - 1, -1, -1):
                if (boxes[i].dimensions.x + padding) > spaces[p][3] or (boxes[i].dimensions.y + padding) > spaces[p][4]: continue
                boxes[i].location.x = spaces[p][0] - (boxes[i].dimensions.x + padding)
                boxes[i].location.y = spaces[p][1] + (boxes[i].dimensions.y + padding)
                if (boxes[i].dimensions.x + padding) == spaces[p][3] and (boxes[i].dimensions.y + padding) == spaces[p][4]:
                    last = spaces.pop()
                    if p < spaces.count: spaces[p] = last
                elif (boxes[i].dimensions.y + padding) == spaces[p][4]:
                    spaces[p][0] += (boxes[i].dimensions.x + padding)
                    spaces[p][3] -= (boxes[i].dimensions.x + padding)
                elif (boxes[i].dimensions.x + padding) == spaces[p][3]:
                    spaces[p][1] += (boxes[i].dimensions.y + padding)
                    spaces[p][4] -= (boxes[i].dimensions.y + padding)
                else:
                    spaces.append([
                        spaces[p][0] - (boxes[i].dimensions.x + padding),
                        spaces[p][1],
                        0.0,
                        spaces[p][3] - (boxes[i].dimensions.x + padding),
                        (boxes[i].dimensions.y + padding)
                        ])
                    spaces[p][1] += (boxes[i].dimensions.y + padding)
                    spaces[p][4] -= (boxes[i].dimensions.y + padding)
                break
        return None
    
    def sort(self):
        self.pack_nodes([n for n in self.data.node_tree.nodes if n.type != 'OUTPUT_MATERIAL'], 0.45, -10)
        for n in self.data.node_tree.nodes:
            #print("%s\t%i\t%i\t%s" % (n.dimensions, n.width, n.height, n.name))
            n.update()
        return None


class fopen:
    little_endian = True
    file = ""
    mode = 'rb'
    data = bytearray()
    size = 0
    pos = 0
    isGood = False

    def __init__(self, filename=None, mode='rb', isLittleEndian=True):
        if mode == 'rb':
            if filename != None and Path(filename).is_file():
                self.data = open(filename, mode).read()
                self.size = len(self.data)
                self.pos = 0
                self.mode = mode
                self.file = filename
                self.little_endian = isLittleEndian
                self.isGood = True
        else:
            self.file = filename
            self.mode = mode
            self.data = bytearray()
            self.pos = 0
            self.size = 0
            self.little_endian = isLittleEndian
            self.isGood = False

        return None

    # def __del__(self):
    #    self.flush()

    def resize(self, dataSize=0):
        if dataSize > 0:
            self.data = bytearray(dataSize)
        else:
            self.data = bytearray()
        self.pos = 0
        self.size = dataSize
        self.isGood = False
        return None

    def flush(self):
        print("flush")
        print("file:\t%s" % self.file)
        print("isGood:\t%s" % self.isGood)
        print("size:\t%s" % len(self.data))
        if self.file != "" and not self.isGood and len(self.data) > 0:
            self.isGood = True

            s = open(self.file, 'w+b')
            s.write(self.data)
            s.close()

    def read_and_unpack(self, unpack, size):
        '''
          Charactor, Byte-order
          @,         native, native
          =,         native, standard
          <,         little endian
          >,         big endian
          !,         network

          Format, C-type,         Python-type, Size[byte]
          c,      char,           byte,        1
          b,      signed char,    integer,     1
          B,      unsigned char,  integer,     1
          h,      short,          integer,     2
          H,      unsigned short, integer,     2
          i,      int,            integer,     4
          I,      unsigned int,   integer,     4
          f,      float,          float,       4
          d,      double,         float,       8
        '''
        value = 0
        if self.size > 0 and self.pos + size <= self.size:
            value = struct.unpack_from(unpack, self.data, self.pos)[0]
            self.pos += size
        return value

    def pack_and_write(self, pack, size, value):
        if self.pos + size > self.size:
            self.data.extend(b'\x00' * ((self.pos + size) - self.size))
            self.size = self.pos + size
        try:
            struct.pack_into(pack, self.data, self.pos, value)
        except:
            print('Pos:\t%i / %i (buf:%i) [val:%i:%i:%s]' % (self.pos, self.size, len(self.data), value, size, pack))
            pass
        self.pos += size
        return None

    def set_pointer(self, offset):
        self.pos = offset
        return None
    
    def set_endian(self, isLittle = True):
        self.little_endian = isLittle
        return isLittle


def fclose(bitStream=fopen()):
    bitStream.flush()
    bitStream.isGood = False


def fseek(bitStream=fopen(), offset = 0, dir = 0):
    if dir == 0:
        bitStream.set_pointer(offset)
    elif dir == 1:
        bitStream.set_pointer(bitStream.pos + offset)
    elif dir == 2:
        bitStream.set_pointer(bitStream.pos - offset)
    return None


def ftell(bitStream=fopen()):
    return bitStream.pos


def readByte(bitStream=fopen(), isSigned=0):
    fmt = 'b' if isSigned == 0 else 'B'
    return (bitStream.read_and_unpack(fmt, 1))


def readShort(bitStream=fopen(), isSigned=0):
    fmt = '>' if not bitStream.little_endian else '<'
    fmt += 'h' if isSigned == 0 else 'H'
    return (bitStream.read_and_unpack(fmt, 2))


def readLong(bitStream=fopen(), isSigned=0):
    fmt = '>' if not bitStream.little_endian else '<'
    fmt += 'i' if isSigned == 0 else 'I'
    return (bitStream.read_and_unpack(fmt, 4))


def readLongLong(bitStream=fopen(), isSigned=0):
    fmt = '>' if not bitStream.little_endian else '<'
    fmt += 'q' if isSigned == 0 else 'Q'
    return (bitStream.read_and_unpack(fmt, 8))


def readFloat(bitStream=fopen()):
    fmt = '>f' if not bitStream.little_endian else '<f'
    return (bitStream.read_and_unpack(fmt, 4))


def readDouble(bitStream=fopen()):
    fmt = '>d' if not bitStream.little_endian else '<d'
    return (bitStream.read_and_unpack(fmt, 8))


def readHalf(bitStream=fopen()):
    uint16 = bitStream.read_and_unpack('>H' if not bitStream.little_endian else '<H', 2)
    uint32 = (
        (((uint16 & 0x03FF) << 0x0D) | ((((uint16 & 0x7C00) >> 0x0A) + 0x70) << 0x17)) |
        (((uint16 >> 0x0F) & 0x00000001) << 0x1F)
        )
    return struct.unpack('f', struct.pack('I', uint32))[0]


def readString(bitStream=fopen(), length=0):
    string = ''
    pos = bitStream.pos
    lim = length if length != 0 else bitStream.size - bitStream.pos
    for i in range(0, lim):
        b = bitStream.read_and_unpack('B', 1)
        if b != 0:
            string += chr(b)
        else:
            if length > 0:
                bitStream.set_pointer(pos + length)
            break
    return string


def writeByte(bitStream=fopen(), value=0):
    bitStream.pack_and_write('B', 1, int(value))
    return None


def writeShort(bitStream=fopen(), value=0):
    fmt = '>H' if not bitStream.little_endian else '<H'
    bitStream.pack_and_write(fmt, 2, int(value))
    return None


def writeLong(bitStream=fopen(), value=0):
    fmt = '>I' if not bitStream.little_endian else '<I'
    bitStream.pack_and_write(fmt, 4, int(value))
    return None


def writeFloat(bitStream=fopen(), value=0.0):
    fmt = '>f' if not bitStream.little_endian else '<f'
    bitStream.pack_and_write(fmt, 4, value)
    return None


def writeLongLong(bitStream=fopen(), value=0):
    fmt = '>Q' if not bitStream.little_endian else '<Q'
    bitStream.pack_and_write(fmt, 8, value)
    return None


def writeDoube(bitStream=fopen(), value=0.0):
    fmt = '>d' if not bitStream.little_endian else '<d'
    bitStream.pack_and_write(fmt, 8, value)
    return None


def writeString(bitStream=fopen(), string="", length=0):
    strLen = len(string)
    if length == 0: length = strLen + 1
    for i in range(0, length):
        if i < strLen:
            bitStream.pack_and_write('b', 1, ord(string[i]))
        else:
            bitStream.pack_and_write('B', 1, 0)
    return None

def mesh_validate (vertices=[], faces=[]):
    #
    # Returns True if mesh is BAD
    #
    # check face index bound
    face_min = 0
    face_max = len(vertices) - 1
    
    for face in faces:
        for side in face:
            if side < face_min or side > face_max:
                print("Face Index Out of Range:\t[%i / %i]" % (side, face_max))
                return True
    return False

def mesh(
    vertices=[],
    faces=[],
    materialIDs=[],
    tverts=[],
    normals=[],
    colours=[],
    materials=[],
    mscale=1.0,
    flipAxis=False,
    obj_name="Object",
    lay_name='',
    position = (0.0, 0.0, 0.0)
    ):
    #
    # This function is pretty, ugly
    # imports the mesh into blender
    #
    # Clear Any Object Selections
    # for o in bpy.context.selected_objects: o.select = False
    bpy.context.view_layer.objects.active = None
    
    # Get Collection (Layers)
    if lay_name != '':
        # make collection
        layer = bpy.data.collections.get(lay_name)
        if layer == None:
            layer = bpy.data.collections.new(lay_name)
            bpy.context.scene.collection.children.link(layer)
    else:
        if len(bpy.data.collections) == 0:
            layer = bpy.data.collections.new("Collection")
            bpy.context.scene.collection.children.link(layer)
        else:
            try:
                layer = bpy.data.collections[bpy.context.view_layer.active_layer_collection.name]
            except:
                layer = bpy.data.collections[0]
    

    # make mesh
    msh = bpy.data.meshes.new('Mesh')

    # msh.name = msh.name.replace(".", "_")

    # Apply vertex scaling
    # mscale *= bpy.context.scene.unit_settings.scale_length
    vertArray=[]
    if len(vertices) > 0:
        vertArray = [[float] * 3] * len(vertices)
        if flipAxis:
            for v in range(0, len(vertices)):
                vertArray[v] = (
                    vertices[v][0] * mscale,
                    -vertices[v][2] * mscale,
                    vertices[v][1] * mscale
                )
        else:
            for v in range(0, len(vertices)):
                vertArray[v] = (
                    vertices[v][0] * mscale,
                    vertices[v][1] * mscale,
                    vertices[v][2] * mscale
                )

    # assign data from arrays
    if mesh_validate(vertArray, faces):
        # Erase Mesh
        msh.user_clear()
        bpy.data.meshes.remove(msh)
        print("Mesh Deleted!")
        return None
    
    msh.from_pydata(vertArray, [], faces)

    # set surface to smooth
    msh.polygons.foreach_set("use_smooth", [True] * len(msh.polygons))

    # Set Normals
    if len(faces) > 0:
        if len(normals) > 0:
            msh.use_auto_smooth = True
            if len(normals) == (len(faces) * 3):
                msh.normals_split_custom_set(normals)
            else:
                normArray = [[float] * 3] * (len(faces) * 3)
                if flipAxis:
                    for i in range(0, len(faces)):
                        for v in range(0, 3):
                            normArray[(i * 3) + v] = (
                                [normals[faces[i][v]][0],
                                 -normals[faces[i][v]][2],
                                 normals[faces[i][v]][1]]
                            )
                else:
                    for i in range(0, len(faces)):
                        for v in range(0, 3):
                            normArray[(i * 3) + v] = (
                                [normals[faces[i][v]][0],
                                 normals[faces[i][v]][1],
                                 normals[faces[i][v]][2]]
                            )
                msh.normals_split_custom_set(normArray)

        # create texture corrdinates
        #print("tverts ", len(tverts))
        # this is just a hack, i just add all the UVs into the same space <<<
        if len(tverts) > 0:
            uvw = msh.uv_layers.new()
            # if len(tverts) == (len(faces) * 3):
            #    for v in range(0, len(faces) * 3):
            #        msh.uv_layers[uvw.name].data[v].uv = tverts[v]
            # else:
            uvwArray = [[float] * 2] * len(tverts[0])
            for i in range(0, len(tverts[0])):
                uvwArray[i] = [0.0, 0.0]

            for v in range(0, len(tverts[0])):
                for i in range(0, len(tverts)):
                    uvwArray[v][0] += tverts[i][v][0]
                    uvwArray[v][1] += 1.0 - tverts[i][v][1]

            for i in range(0, len(faces)):
                for v in range(0, 3):
                    msh.uv_layers[uvw.name].data[(i * 3) + v].uv = (
                        uvwArray[faces[i][v]][0],
                        uvwArray[faces[i][v]][1]
                    )

        # create vertex colours
        if len(colours) > 0:
            col = msh.vertex_colors.new()
            if len(colours) == (len(faces) * 3):
                for v in range(0, len(faces) * 3):
                    msh.vertex_colors[col.name].data[v].color = colours[v]
            else:
                colArray = [[float] * 4] * (len(faces) * 3)
                for i in range(0, len(faces)):
                    for v in range(0, 3):
                        msh.vertex_colors[col.name].data[(i * 3) + v].color = colours[faces[i][v]]
        else:
            # Use colours to make a random display
            col = msh.vertex_colors.new()
            random_col = rancol4()
            for v in range(0, len(faces) * 3):
                msh.vertex_colors[col.name].data[v].color = random_col

    # Create Face Maps?
    # msh.face_maps.new()

    # Check mesh is Valid
    # Without this blender may crash!!! lulz
    # However the check will throw false positives so
    # an additional or a replacement valatiation function
    # would be required
    
    if msh.validate(clean_customdata=False):
        print("Mesh Failed Validation")

    # Update Mesh
    msh.update()

    # Assign Mesh to Object
    obj = bpy.data.objects.new(obj_name, msh)
    obj.location = position
    # obj.name = obj.name.replace(".", "_")

    for i in range(0, len(materials)):
        if len(obj.material_slots) < (i + 1):
            # if there is no slot then we append to create the slot and assign
            if type(materials[i]).__name__ == 'StandardMaterial':
                obj.data.materials.append(materials[i].data)
            else:
                obj.data.materials.append(materials[i])
        else:
            # we always want the material in slot[0]
            if type(materials[i]).__name__ == 'StandardMaterial':
                obj.material_slots[0].material = materials[i].data
            else:
                obj.material_slots[0].material = materials[i]
        # obj.active_material = obj.material_slots[i].material

    if len(materialIDs) == len(obj.data.polygons):
        for i in range(0, len(materialIDs)):
            obj.data.polygons[i].material_index = materialIDs[i]
            if materialIDs[i] > len(materialIDs):
                materialIDs[i] = materialIDs[i] % len(materialIDs)
            
    elif len(materialIDs) > 0:
        print("Error:\tMaterial Index Out of Range")

    # obj.data.materials.append(material)
    layer.objects.link(obj)

    # Generate a Material
    # img_name = "Test.jpg"  # dummy texture
    # mat_count = len(texmaps)

    # if mat_count == 0 and len(materialIDs) > 0:
    #    for i in range(0, len(materialIDs)):
    #        if (materialIDs[i] + 1) > mat_count: mat_count = materialIDs[i] + 1

    # Assign Material ID's
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.context.tool_settings.mesh_select_mode = [False, False, True]

    bpy.ops.object.mode_set(mode='OBJECT')
    # materialIDs

    # Redraw Entire Scene
    # bpy.context.scene.update()

    return obj






#
# ====================================================================================
# FORMAT STRCTURES FOR Rebel Galaxy Outlaw
# ====================================================================================
# These describe the mdl file format
# ====================================================================================
#


class fmtRGOMDL_String:
    """string[] 	"""                     
    value           = ""
    """uint8_t  	"""                     
    type            = 1
    """uint16_t	"""                         
    unk003          = 0
    """uint16_t	"""                         
    unk004          = 0
    def read (self, f=fopen()):
        b = 0
        s = ""
        p = ftell(f)
        while b != None:
            
            b = readByte(f)
            if b > 31 and b != None:
                s += chr(b)
                p = ftell(f)
            else:
                break
            
        fseek(f, p, seek_set)
        self.type = readByte(f)
        if self.type == 0x01:
            self.unk003 = readByte(f)
        elif self.type == 0x0A:
            self.unk003 = readShort(f)
            self.unk004 = readShort(f)
        self.value = s
        return s
        
    

class fmtRGOMDL_Vertex_Def: # 10 Bytes
    """uint16_t	"""
    map                 = 0 # channel?
    """uint16_t	"""
    datatype            = 0 # data type?
    """uint16_t	"""
    vertextype          = 0 # component type?
    """uint16_t	"""
    vertexpos           = 0 # component position
    """uint16_t	"""
    unk006              = 0 # ?
    def read (self, f=fopen()):
        self.map                                   = readShort(f)
        self.datatype                              = readShort(f)
        self.vertextype                            = readShort(f)
        self.vertexpos                             = readShort(f)
        self.unk006                                = readShort(f)
        return None
    

class fmtRGOMDL_Table2:
    
    indicies1 = [] # 32 shorts
    unk016 = [[float], [float], [float]]
    unk017 = [[float], [float], [float]]
    indicies2 = [] # 12 longs
    mat_name = fmtRGOMDL_String()
    
    
    def read (self, f=fopen(), headerName=fmtRGOMDL_String()):
        self.indicies1 = [int] * 32
        for i in range(0, 32):
            self.indicies1[i] = readShort(f)
            
        self.unk016 = [readFloat(f), readFloat(f), readFloat(f)]
        self.unk017 = [readFloat(f), readFloat(f), readFloat(f)]
        if headerName.unk003 == 14:
            fseek(f, 20, seek_cur)
            
        self.indicies2 = [int] * 12
        for i in range(0, 12):
            self.indicies2[i] = readLong(f)
            
        mat_name = fmtRGOMDL_String()
        self.mat_name.read(f)
        return None
    

class fmtRGOMDL_Table9:
    unk034 = 0
    count = 0
    indices3 = []
    def read (self, f=fopen()):
        self.indices3 = []
        self.unk034 = readShort(f)
        self.count = readLong(f) 
        if self.count > 0:
            count_tmp = int(self.count * 3)
            self.indices3 = [int] * count_tmp
            for i in range(0, count_tmp):
                self.indices3[i]  = readLong(f)
                
        return None
        
    

class fmtRGOMDL:
    """uint8_t  	"""
    unk001 = 0
    """uint8_t  	"""
    unk002 = 0x10
    """fmtRGOMDL_String"""
    fileVer = fmtRGOMDL_String()
    """uint16_t	"""
    unk007 = 0
    """uint16_t	"""
    unk008 = 0
    """uint16_t	"""
    fvf_count = 0
    """fmtRGOMDL_Vertex_Def[]"""
    fvf_table = []
    """float    	"""
    unk009 = 0.0
    """uint32_t	"""
    unk010 = 0
    """uint32_t	"""
    unk011 = 0
    """uint32_t	"""
    unk012 = 0
    """uint32_t	"""
    unk013 = 0
    """uint32_t	"""
    table2_count = 0
    """uint32_t	"""
    texture_count = 0
    """string[]	"""
    texture_array = []
    """fmtRGOMDL_String"""
    model_name = fmtRGOMDL_String()
    table2 = []
    position = [[float], [float], [float]]
    array1 = [] # Indices
    array2 = [] # Vertices
    array3 = [] # Normals?
    array4 = [] # Tagents?
    value5 = 0
    array6 = [] # TVert0[]
    array7 = [] # TVert1[]
    array8 = [] # Vertices
    array5 = [] # Indices
    unk022 = 0
    unk023 = 0
    unk024 = 0
    unk025 = 0
    unk026 = 0
    array9 = [] # Indices[]?
    
    def read (self, f=fopen()):
        
        pos = ftell(f)
        fileType = readLong(f)
        
        if fileType == 0x4D5B1000:
            fseek(f, pos, seek_set)
            self.unk001 = readByte(f)
            self.unk002 = readByte(f)
            self.fileVer.read(f)
            
            if self.fileVer.value == "[MeshSerializer_Runic]":
                self.unk007 = readShort(f)
                self.unk008 = readShort(f)
                self.fvf_count = readShort(f)
                self.fvf_table = []
                if self.fvf_count > 0:
                    self.fvf_table = [fmtRGOMDL_Vertex_Def] * self.fvf_count
                    for i in range(0, self.fvf_count):
                        self.fvf_table[i] = fmtRGOMDL_Vertex_Def()
                        self.fvf_table[i].read(f)
                        
                    
                self.unk009 = readFloat(f)
                self.unk010 = readLong(f)
                self.unk011 = readLong(f)
                self.unk012 = readLong(f)
                self.unk013 = readLong(f)
                self.table2_count = readLong(f)
                self.texture_count = readLong(f)
                self.texture_array = []
                if self.texture_count > 0:
                    self.texture_array = [fmtRGOMDL_String] * self.texture_count
                    for i in range(0, self.texture_count):
                        self.texture_array[i] = fmtRGOMDL_String()
                        self.texture_array[i].read(f)
                        
                    
                self.model_name.read(f)
                self.table2 = []
                if self.table2_count > 0:
                    self.table2 = [fmtRGOMDL_Table2] * self.table2_count
                    for i in range(0, self.table2_count):
                        self.table2[i] = fmtRGOMDL_Table2()
                        self.table2[i].read(f, self.fileVer)
                        
                    
                self.position = [readFloat(f), readFloat(f), readFloat(f)]
                v = 1
                count1 = 0
                count2 = 0
                count3 = 0
                count5 = 0
                count6 = 0
                self.array6_tmp = []
                self.array7_tmp = []
                self.value5 = 0
                self.array1 = []
                self.array2 = []
                self.array3 = []
                self.array4 = []
                self.array5 = []
                self.array6 = []
                self.array7 = []
                for i in range(0, self.fvf_count):
                    
                    if self.fvf_table[i].vertextype == 1:	# Faces?
                        count1 = readLong(f)
                        if count1 > 0:
                            self.array1 = [int] * count1
                            for v in range(0, count1):
                                self.array1[v] = readLong(f)
                                
                            
                        
                    elif self.fvf_table[i].vertextype == 9:	# Positions?
                        count2 = readLong(f)
                        if count2 > 0:
                            self.array2 = [[float] * 3] * count2
                            for v in range(0, count2):
                                self.array2[v] = [readFloat(f), readFloat(f), readFloat(f)]
                                
                            
                        
                    elif self.fvf_table[i].vertextype == 8:	# Normal?
                        count3 = readLong(f)
                        if count3 > 0:
                            self.array3 = [[float] * 3] * count3
                            for v in range(0, count3):
                                self.array3[v] = [readFloat(f), readFloat(f), readFloat(f)]
                                
                            
                        
                    elif self.fvf_table[i].vertextype == 4:	# ??
                        
                        if count3 > 0:
                            self.array4 = [[float] * 3] * count3
                            for v in range(0, count3):
                                self.array4[v] = [readFloat(f), readFloat(f), readFloat(f)]
                                
                            
                        self.value5 = readShort(f) # ? type
                        
                    elif self.fvf_table[i].vertextype == 7:	# 0 	1 	7 	52 	0 vert 12
                        if count3 > 0:
                            if count1 > 0:
                                self.array7_tmp = [int] * count1
                                for v in range(0, count1):
                                    self.array7_tmp[v] = readLong(f)
                                    
                                self.array7.append(self.array7_tmp)
                                
                            count6 = readLong(f)
                            if count6 > 0:
                                self.array6_tmp = [[float] * 3] * count6
                                for v in range(0, count6):
                                    self.array6_tmp[v] = [readFloat(f), readFloat(f), 0.0]
                                    
                                self.array6.append(self.array6_tmp)
                                
                            
                        
                    else:
                        print("Error: \tUnknown Vertex Type: \t%i\n" %(self.fvf_table[i].vertextype))
                        
                    
                
                self.unk022 = readShort(f) # ? type
                self.array8 = []
                self.array5 = []
                if count3 > 0:
                    if count1 > 0:
                        self.array5 = [int] * count1
                        for i in range(0, count1):
                            self.array5[i] = readLong(f)
                            
                        
                    count8 = readLong(f)
                    if count8 > 0:
                        self.array8 = [[float] * 3] * count8
                        for i in range(0, count8):
                            self.array8[i] = [readFloat(f), readFloat(f), readFloat(f)]
                            
                        
                    self.unk023 = readLong(f) # ?
                    
                self.unk024 = readLong(f) # 0
                self.unk025 = readLong(f) # index table count
                self.unk026 = readLong(f)  # how many index tables
                if self.unk026 > 0:
                    self.array9 = []
                    self.array9 = [fmtRGOMDL_Table9] * self.unk026
                    for i in range(0, self.unk026):
                        self.array9[i] = fmtRGOMDL_Table9()
                        self.array9[i].read(f)
                        
                    
                
            else:
                print("error: \tUnsupported file type %s\n" %(self.fileVer))
                
            
            
        else:
            print("error: \tinvalid file type\n")
        



#
# ====================================================================================
# MAIN FUNCTION
# ====================================================================================
# function used in the main operation of the script
# ====================================================================================
#


def read(file="", armName="Armature", mscale=0.1):
    clearListener()
    
    f = fopen(file, 'rb')
    fpath = ""
    if f.isGood:

        # Read File into YOBJ Class
        mdl = fmtRGOMDL()
        mdl.read(f)
        fpath = getFilenamePath(file)
        fclose(f)
        
        
        array1_count = len(mdl.array1)
        array9_count = len(mdl.array9)
        faceArray = []
        vertArray = []
        if array1_count > 0:
            for x in range(0, array9_count):

                for i in range(0, mdl.array9[x].count):
                    faceArray.append([mdl.array9[x].indices3[(i * 3) + 0], mdl.array9[x].indices3[(i * 3) + 1], mdl.array9[x].indices3[(i * 3) + 2]])
                    
                
            array2_count = len(mdl.array2)
            vertArray = [[float] * 3] * array1_count
            for i in range(0, array1_count):
                vertArray[i] = [mdl.array2[mdl.array1[i]][0] * mscale, mdl.array2[mdl.array1[i]][2] * mscale, mdl.array2[mdl.array1[i]][1] * mscale]
                
        
        msh = mesh (vertices=vertArray, faces=faceArray)
        
    
        
        
        
        
        
        print("Done")
    
    
    return None
    


#
# ====================================================================================
# BLENDER API FUNCTIONS
# ====================================================================================
# These are functions or wrappers specific with dealing with blenders API
# ====================================================================================
#

# Callback when file(s) are selected
def wrapper1_callback(fpath="", files=[], clearScene=True, armName="Armature", mscale=1.0):
    if len(files) > 0 and clearScene: deleteScene(['MESH', 'ARMATURE'])
    for file in files:
        read (fpath + file.name, armName, mscale)
    if len(files) > 0:
        messageBox("Done!")
        return True
    else:
        return False


# Define Operator
class ImportHelper_wrapper1(bpy.types.Operator):

    # Operator Path
    bl_idname = "importhelper.wrapper1"
    bl_label = "Select File"
    filename_ext = ".mdl"

    # Operator Properties
    # filter_glob: bpy.props.StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'})
    filter_glob: bpy.props.StringProperty(default='*.mdl', options={'HIDDEN'}, subtype='FILE_PATH')

    # Variables
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # full path of selected item (path+filename)
    filename: bpy.props.StringProperty(subtype="FILE_NAME")  # name of selected item
    directory: bpy.props.StringProperty(subtype="FILE_PATH")  # directory of the selected item
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement)  # a collection containing all the selected itemsï¿½f filenames

    # Controls
    #my_int1: bpy.props.IntProperty(name="Some Integer", description="Tooltip")
    my_float1: bpy.props.FloatProperty(name="Scale", default=0.1, description="Changes Scale of the imported Mesh")
    # my_float2: bpy.props.FloatProperty(name="Some Float point", default = 0.25, min = -0.25, max = 0.5)
    my_bool1: bpy.props.BoolProperty(name="Clear Scene", default=False, description="Deletes everything in the scene prior to importing")
    #my_bool2: bpy.props.BoolProperty(name="Skeleton", default=False, description="Imports Bones to an Armature")
    #my_bool3: bpy.props.BoolProperty(name="Vertex Weights", default=False, description="Builds Vertex Groups")
    #my_bool4: bpy.props.BoolProperty(name="Vertex Normals", default=False, description="Applies Custom Normals")
    #my_bool5: bpy.props.BoolProperty(name="Vertex Colours", default=False, description="Builds Vertex Colours")
    #my_bool6: bpy.props.BoolProperty(name="Guess Parents", default=False, description="Uses algorithm to Guess Bone Parenting")
    #my_bool7: bpy.props.BoolProperty(name="Dump Textures", default=False, description="Writes Textures from a file pair '_tex.bin'")
    my_string1: bpy.props.StringProperty(name="", default="Armature", description="Name of Armature to Import Bones to")
    #my_dropdown1: bpy.props.EnumProperty(
    #    name="Drop",
    #    items=[
    #        ('CLEAR', 'clear scene', 'clear scene'),
    #        ('ADD_CUBE', 'add cube', 'add cube'),
    #        ('ADD_SPHERE', 'add sphere', 'add sphere')
    #    ]
    #)
    #my_dropdown2: bpy.props.EnumProperty(
    #    name="Drop",
    #    items=[
    #        ('CLEAR', 'clear scene', 'clear scene'),
    #        ('ADD_CUBE', 'add cube', 'add cube'),
    #        ('ADD_SPHERE', 'add sphere', 'add sphere')
    #    ]
    #)

    # Runs when this class OPENS
    def invoke(self, context, event):

        # Retrieve Settings
        try:
            self.filepath = bpy.types.Scene.wrapper1_filepath
        except:
            bpy.types.Scene.wrapper1_filepath = bpy.props.StringProperty(subtype="FILE_PATH")

        try:
            self.directory = bpy.types.Scene.wrapper1_directory
        except:
            bpy.types.Scene.wrapper1_directory = bpy.props.StringProperty(subtype="FILE_PATH")

        try:
            self.my_float1 = bpy.types.Scene.wrapper1_my_float1
        except:
            bpy.types.Scene.wrapper1_my_float1 = bpy.props.FloatProperty(default=0.1)

        try:
            self.my_bool1 = bpy.types.Scene.wrapper1_my_bool1
        except:
            bpy.types.Scene.wrapper1_my_bool1 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool2 = bpy.types.Scene.wrapper1_my_bool2
        #except:
        #    bpy.types.Scene.wrapper1_my_bool2 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool3 = bpy.types.Scene.wrapper1_my_bool3
        #except:
        #    bpy.types.Scene.wrapper1_my_bool3 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool4 = bpy.types.Scene.wrapper1_my_bool4
        #except:
        #    bpy.types.Scene.wrapper1_my_bool4 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool5 = bpy.types.Scene.wrapper1_my_bool5
        #except:
        #    bpy.types.Scene.wrapper1_my_bool5 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool6 = bpy.types.Scene.wrapper1_my_bool6
        #except:
        #    bpy.types.Scene.wrapper1_my_bool6 = bpy.props.BoolProperty(default=False)

        #try:
        #    self.my_bool7 = bpy.types.Scene.wrapper1_my_bool7
        #except:
        #    bpy.types.Scene.wrapper1_my_bool7 = bpy.props.BoolProperty(default=False)

        try:
            self.my_string1 = bpy.types.Scene.my_string1
        except:
            bpy.types.Scene.my_string1 = bpy.props.BoolProperty(default=False)

        # Open File Browser
        # Set Properties of the File Browser
        context.window_manager.fileselect_add(self)
        context.area.tag_redraw()

        return {'RUNNING_MODAL'}

    # Runs when this Window is CANCELLED
    def cancel(self, context):
        print("run bitch")

    # Runs when the class EXITS
    def execute(self, context):

        # Save Settings
        bpy.types.Scene.wrapper1_filepath = self.filepath
        bpy.types.Scene.wrapper1_directory = self.directory
        bpy.types.Scene.wrapper1_my_float1 = self.my_float1
        bpy.types.Scene.wrapper1_my_bool1 = self.my_bool1
        #bpy.types.Scene.wrapper1_my_bool2 = self.my_bool2
        #bpy.types.Scene.wrapper1_my_bool3 = self.my_bool3
        #bpy.types.Scene.wrapper1_my_bool4 = self.my_bool4
        #bpy.types.Scene.wrapper1_my_bool5 = self.my_bool5
        #bpy.types.Scene.wrapper1_my_bool6 = self.my_bool6
        #bpy.types.Scene.wrapper1_my_bool7 = self.my_bool7
        bpy.types.Scene.wrapper1_my_string1 = self.my_string1

        # Run Callback
        wrapper1_callback(
            self.directory + "\\",
            self.files,
            self.my_bool1,    # ClearScene
            #self.my_bool2,
            self.my_string1,  # Armature Name
            #self.my_bool4,
            #self.my_bool5,
            #self.my_bool3,
            #self.my_bool6,
            #self.my_bool7,
            self.my_float1    # Mesh Scale
        )

        return {"FINISHED"}

        # Window Settings

    def draw(self, context):

        # Set Properties of the File Browser
        # context.space_data.params.use_filter = True
        # context.space_data.params.use_filter_folder=True #to not see folders

        # Configure Layout
        # self.layout.use_property_split = True       # To Enable Align
        # self.layout.use_property_decorate = False   # No animation.

        self.layout.row().label(text="Import Settings")

        self.layout.separator()
        self.layout.row().prop(self, "my_bool1")
        self.layout.row().prop(self, "my_float1")

        #box = self.layout.box()
        #box.label(text="Include")
        #box.prop(self, "my_bool2")
        #box.prop(self, "my_bool3")
        #box.prop(self, "my_bool4")
        #box.prop(self, "my_bool5")

        box = self.layout.box()
        #box.label(text="Misc")
        #box.prop(self, "my_bool6")
        #box.prop(self, "my_bool7")
        box.label(text="Import Bones To:")
        box.prop(self, "my_string1")

        self.layout.separator()

        col = self.layout.row()
        col.alignment = 'RIGHT'
        col.label(text="  Author:", icon='QUESTION')
        col.alignment = 'LEFT'
        col.label(text="mariokart64n")

        col = self.layout.row()
        col.alignment = 'RIGHT'
        col.label(text="Release:", icon='GRIP')
        col.alignment = 'LEFT'
        col.label(text="August 08, 2022")

    def menu_func_import(self, context):
        self.layout.operator("importhelper.wrapper1", text="[PC] Rebel Galaxy Outlaw (*.mdl)")






#
# ====================================================================================
# BLENDER REGISTRATION
# ====================================================================================
# Registers the script so that it can be called through the import dialog
# ====================================================================================
#

bl_info = {
    "name": "[PC] Rebel Galaxy Outlaw",
    "author": "mariokart64n",
    "version": (1, 0),
    "blender": (3, 2, 2),
    "location": "File > Import",
    "description": "Imports Geometry from .mdl files",
    "warning": "",
    "wiki_url": "https://forum.xentax.com/viewtopic.php?p=186362#p186362",
    "category": "Import-Export",
    }


# Register Operator
def register():
    
    bpy.utils.register_class(ImportHelper_wrapper1)
    #bpy.utils.register_class(ImportHelper_wrapper1)
    #bpy.utils.register_class(
    #    bpy.types.Operator.bl_rna_get_subclass_py('IMPORTHELPER_OT_wrapper1')
    #    )
    #bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    #bpy.types.TOPBAR_MT_file_import.append(ImportHelper_wrapper1.menu_func_import)
    bpy.types.TOPBAR_MT_file_import.append(ImportHelper_wrapper1.menu_func_import)
    #bpy.types.TOPBAR_MT_file_import.append (
    #    bpy.types.Operator.bl_rna_get_subclass_py('IMPORTHELPER_OT_wrapper1').menu_func_import
    #    )
    # Assign Shortcut key
    # bpy.context.window_manager.keyconfigs.active.keymaps["Window"].keymap_items.new('bpy.ops.text.run_script()', 'E', 'PRESS', ctrl=True, shift=False, repeat=False)
    

    return None

# Un- Register Operator
def unregister():

    try:
        bpy.types.TOPBAR_MT_file_import.remove (
            bpy.types.Operator.bl_rna_get_subclass_py('IMPORTHELPER_OT_wrapper1').menu_func_import
            )
    except:
        print("Failed to Unregister2")

    try:
        bpy.utils.unregister_class(
            bpy.types.Operator.bl_rna_get_subclass_py('IMPORTHELPER_OT_wrapper1')
            )
    except:
        print("Failed to Unregister1")
    try:
        bpy.utils.unregister_class(ImportHelper_wrapper1)
    except:
        print("Failed to Unregister3")

    return None


if __name__ == "__main__":
    if not useOpenDialog:

        deleteScene(['MESH', 'ARMATURE'])  # Clear Scene
        read (
            "C:\\Users\\Corey\\Desktop\\rgo-mdl-files\\1_GUITAR.MDL"
            )
        messageBox("Done!")
    else:
        
        # Un-Register Operator
        unregister()
        
        # Register Operator
        register()
        
        # Call ImportHelper
        bpy.ops.importhelper.wrapper1('INVOKE_DEFAULT')