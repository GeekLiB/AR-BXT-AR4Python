import cv2
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pygame, pygame.image
from pygame.locals import *

import BAR4Py.getPMatrix as getPMatrix
import BAR4Py.objloader as objloader

def getGLPM(markImage, sceneImage, isMatches):
    height, width = sceneImage.shape[:2]

    # Init PM.
    pm = getPMatrix.GetPMatrix(markImage)
    # Get dst.
    dst = None
    if isMatches:
        dst = pm.getMatches(sceneImage)
    else:
        dst = pm.findMark(sceneImage)
    if dst is None:
        exit()
    print dst
    # Get ret, mtx, dist, rvecs, tvecs
    tmp = pm.getP(dst)
    if tmp is None:
        exit()
    mtx, _, rvec, tvec = tmp
    # Debug code.
    print 'mtx:\n',mtx,'\nrvec:\n',rvec,'\ntvec:\n',tvec

    glP = pm.getGLP(width, height)
    glM = pm.getGLM()
    # Debug code.
    print 'glP:\n',glP,'\nglM:\n',glM

    return glP, glM

def set_projection_from_camera(glP):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glLoadMatrixf(glP)

def set_modelview_from_camera(glM, scale=1.):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glLoadMatrixf(glM)
    glTranslate(0.5,0.5,-0.5)
    glRotate(180, 1, 0, 0)
    glRotate(180, 0, 0, 1)
    glScalef(scale,scale,scale)

def draw_background(imgName):
    bg_image = pygame.image.load(imgName).convert()
    bg_data = pygame.image.tostring(bg_image, 'RGBX', 1)
    width, height = bg_image.get_size()
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_TEXTURE_2D)
    glGT = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, glGT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glBegin(GL_QUADS)
    glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
    glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
    glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
    glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
    glEnd()

    glDeleteTextures(1)

    return glGT

def load_and_draw_model(filName):
    glLightfv(GL_LIGHT0, GL_POSITION,  (-50, 200, 250, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))

    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    obj = objloader.OBJ(filName, swapyz=True)
    glCallList(obj.gl_list)

    return obj

class BAR4Py:
    def __init__(self, captionStr, markImageName, sceneImageName, OBJFileName, isMatches=False):
        markImage = cv2.imread(markImageName)
        sceneImage = cv2.imread(sceneImageName)
        height, width = sceneImage.shape[:2]

        # Init pygame.
        pygame.init()
        pygame.display.set_mode((width,height), OPENGL | DOUBLEBUF)
        pygame.display.set_caption(captionStr)

        draw_background(sceneImageName)

        glP, glM = getGLPM(markImage, sceneImage, isMatches)
        set_projection_from_camera(glP)
        if isMatches:
            set_modelview_from_camera(glM, 0.5)
        else:
            set_modelview_from_camera(glM)

        load_and_draw_model(OBJFileName)

    def run(self):
        while True:
            event = pygame.event.poll()
            if event.type in (QUIT, KEYDOWN):
                break

            pygame.display.flip()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        if sys.argv[1] == 'mark':
            bar4py = BAR4Py('BAR4Py Demo.', './mark.png', './mark_in_scene.png', './box.obj')
        elif sys.argv[1] == 'matches':
            bar4py = BAR4Py('BAR4Py Demo.', './clock.png', './clock_in_scene.png', './hj.obj', True)
        bar4py.run()
