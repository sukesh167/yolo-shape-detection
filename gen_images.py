#!/usr/bin/env python3
import os
from random import random, seed, uniform, choice
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='generate images')
parser.add_argument('--input', dest = 'SHAPE_FOLDER', type=str)
parser.add_argument('--out_dims', dest='M', type=int, default=1024)
parser.add_argument('--nout', dest='N', type=int, default=1000)
parser.add_argument('--labels', action='store_true')

args = parser.parse_args()

M = args.M
N = args.N
SHAPE_FOLDER = args.SHAPE_FOLDER

DEFAULT_SIDE = 64
DEFAULT_SAFETY_MARGIN = DEFAULT_SIDE * (2**0.5)
ROT = 90
MAX_SQUARES = int((M+2*DEFAULT_SAFETY_MARGIN)/DEFAULT_SIDE*0.75)

def get_func_deg1(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:
        return None
    a = (y0 - y1)/(x0 - x1)
    b = y0 - x0 * a
    return lambda x: a * x + b


def is_point_in_square(p, sq):
    x, y = p
    p0, p1, p2, p3 = sq
    side_func0 = get_func_deg1(p0, p1)
    side_func1 = get_func_deg1(p1, p2)
    side_func2 = get_func_deg1(p2, p3)
    side_func3 = get_func_deg1(p3, p0)
    if not side_func0 or not side_func1 or not side_func2 or not side_func3:
        xmin = min(p0[0], p2[0])
        xmax = max(p0[0], p2[0])
        ymin = min(p0[1], p2[1])
        ymax = max(p0[1], p2[1])
        return xmin <= x <= xmax and ymin <= y <= ymax
    return ((y - side_func0(x)) * (y - side_func2(x))) <= 0 and \
           ((y - side_func1(x)) * (y - side_func3(x))) <= 0


def squares_overlap(square0, square1):
    for p0 in square0:
        if is_point_in_square(p0, square1):
            return True
    for p1 in square1:
        if is_point_in_square(p1, square0):
            return True
    xc0 = (square0[0][0] + square0[2][0]) / 2
    yc0 = (square0[0][1] + square0[2][1]) / 2
    if is_point_in_square((xc0, yc0), square1):
        return True

    return False

def generate_random_square(minx=0, miny=0, maxx=M, maxy=M, safety_margin=DEFAULT_SAFETY_MARGIN, side=DEFAULT_SIDE, squares_to_avoid=()):
    while True:
        restart = False

        x0 = safety_margin + random() * (maxx - minx - 2 * safety_margin)
        y0 = safety_margin + random() * (maxy - miny - 2 * safety_margin)

        
        x1 = x0 + side
        y1 = y0

        x2 = x0 + side 
        y2 = y0 + side

        x3 = x0
        y3 = y0 + side

        sqr = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        for square in squares_to_avoid:
            if squares_overlap(sqr, square):
                restart = True
        if restart:
            continue

        return sqr

def rotate_image(rot_img, angle, side = DEFAULT_SIDE):
    centre = side/2 
  
    rot_mat = cv2.getRotationMatrix2D((centre, centre), angle, 1.0)

    rot_mat[0][2] += (side/2) - centre
    rot_mat[1][2] += (side/2) - centre

    img = cv2.warpAffine(rot_img, rot_mat, (side,side))
  
    return(img)

def generator(shapes):
    seed()

    squares = []
    bboxes = []
    img = np.zeros((M, M, 3))
    for _ in range(MAX_SQUARES):
        square = generate_random_square(squares_to_avoid=squares)
        squares.append(square)
        
        angle = random() * ROT
        scale = uniform(0.75, 1)
        x, y = square[0]
        x = int(x)
        y = int(y)
        s = DEFAULT_SIDE
        shape_name = choice(shapes)
        shape = cv2.imread(SHAPE_FOLDER+shape_name)
        s = int(s*scale)
        shape = cv2.resize(shape, (s, s), scale, scale, cv2.INTER_LINEAR)
        shape = rotate_image(shape, angle, s)
        img[x:x+s, y:y+s, :] = shape
        y, x = np.mean(np.array(square), axis=0)
        bboxes.append([shape_name, [x, y, s, s]])
    return(img, bboxes)
if __name__ == "__main__":
    shapes = {k:str(v) for v,k in enumerate(os.listdir(SHAPE_FOLDER))}
    try:
        os.mkdir('imgs')
        if args.labels:
            os.mkdir('labels')
    except:
        pass
    for _ in range(N):
        img, bboxes = generator(list(shapes.keys()))
        cv2.imwrite("imgs/{:06}_{}.jpg".format(_, M), img)
        if args.labels:
            f_l = open("labels/{:06}_{}.txt".format(_, M), "w")
            for bbox in bboxes:
                f_l.write(shapes[bbox[0]]+" "+" ".join([str(temp/M) for temp in bbox[1]])+"\n")
            f_l.close()
    #if args.labels:
    #    f = open("labels/obj.names".format(_), "w")
    #    for k in shapes.keys():
    #       f.write(k[:-4]+"\n")
    #    f.close