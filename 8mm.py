#!/usr/bin/env python3

import cv2
import os
import imutils
import numpy
from math import sqrt
import argparse

parser = argparse.ArgumentParser(description='Analyze 8mm captures')
parser.add_argument('--source', type=str, help="source directory",required=True)
parser.add_argument('--prefix', type=str, help="source filename prefix")
parser.add_argument('--project', type=str, help="project name",required=True)
parser.add_argument('--grayscale',action="store_true",default=False)
args = parser.parse_args()

print(args)

base = args.source
rawname = args.prefix
projectname = args.project
grayscale=args.grayscale

verbose = 0

def check_contour(c, count):
    M = cv2.moments(c)
    cx = int((M["m10"] / M["m00"]))
    cy = int((M["m01"] / M["m00"]))

    rect = cv2.boundingRect(c)
    w = rect[2]
    h = rect[3]

    # find the hole in the correct position and size
    if w < 300 or w > 330:
        if verbose: print(f'reject {count} Wrong width {w}')

        return ( False, cx, cy )
    if h < 210 or h > 230:
        if verbose: print(f'reject {count} wrong height {h}')
        return ( False, cx, cy )
    if cy < 640:
        if verbose: print(f'reject {count} too close to top {cy}')
        return ( False, cx, cy )
    if cx < 1000:
        if verbose: print(f'reject {count} in the wrong position {cx}')
        return ( False, cx, cy )

    return ( True, cx, cy )

def crop8mm(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    ret,thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY_INV)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = [x for x in contours if len(x) > 100]

    count = -1

    centers = []

    for c in cnt:
        count = count+1
        ret, cx, cy = check_contour(c, count)
        centers.append((cx, cy))
        if not ret:
            continue

        rect = ( cx-1035, cy - 655, cx - 135, cy + 15 )
        # cv2.rectangle(img, ( rect[0], rect[1] ), ( rect[2], rect[3] ), (255, 0, 0), 2)
        # ww = rect[2] - rect[0]
        # hh = rect[3] - rect[1]
        # print(f'{cx},{cy} {rect} {ww} {hh}')
        out = gray if grayscale else img

        out = imutils.rotate(out, -0.82, center=( cx, cy ))
        return cv2.flip(out[rect[1]:rect[3], rect[0]:rect[2]], 1)

    # if we got here, maybe try joining controus?
    for a in range(0, len(centers)-1):
        for b in range(a+1, len(centers)):
            aa=centers[a]
            bb=centers[b]
            xx = aa[0] - bb[0]
            yy = aa[1] - bb[1]
            dist = sqrt(xx**2 + yy**2)
            if verbose:
                print(a, b, dist, aa, bb)
            if dist < 200:
                ret, cx, cy = check_contour(numpy.concatenate((cnt[a],cnt[b])), f'{a},{b}')
                if not ret:
                    continue

                rect = ( cx-1035, cy - 655, cx - 135, cy + 15 )
                # cv2.rectangle(img, ( rect[0], rect[1] ), ( rect[2], rect[3] ), (255, 0, 0), 2)
                # ww = rect[2] - rect[0]
                # hh = rect[3] - rect[1]
                # print(f'{cx},{cy} {rect} {ww} {hh}')

                gray = imutils.rotate(gray, -0.82, center=( cx, cy ))
                return cv2.flip(gray[rect[1]:rect[3], rect[0]:rect[2]], 1)

    if verbose:
        img = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, cnt, -1, (0,0,255), 3)
        count = 0
        for c in centers:
            cv2.putText(img, f'{count}', c, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
            count = count+1
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return None

    # 86, 158, 976, 812  1117 799
directory = os.listdir(base)
def keyfunc(name):
    t=name.split('-')[-1].split('.')[0]
    return int(t)
directory.sort(key=keyfunc)

if verbose:
    broken=verbose
    directory=[]
    for i in range(broken-5, broken):
        directory.append(f'{rawname}-{i:04d}.jpg')

skipped = 0
for i in directory:
    fn = base + i
    num = keyfunc(i)
    if num < 0:
        continue
    if verbose:
        print(f'Processing {num} {fn}')
    try:
        img = crop8mm(cv2.imread(fn))
    except Exception as e:
        print(f'Failed to open {fn}: {e}')
        exit(1)
    if img is not None:
        skipped = 0
        oname = f'{projectname}-{num:05d}.jpg'
        cv2.imwrite(oname, img)
        print(f'Found a frame in {oname}')
    else:
        skipped = skipped+1
        if skipped > 10 and False:
            print(f'Run detected at {num}')
            break
