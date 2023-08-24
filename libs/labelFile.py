# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
import os.path
import sys
import math
import numpy as np

class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            difficult = int(shape['difficult'])           
            direction = shape['direction']
            isRotated = shape['isRotated']
            # if shape is normal box, save as bounding box 
            # print('direction is %lf' % direction)
            if not isRotated:
                bndbox = LabelFile.convertPoints2BndBox(points)
                writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], 
                    bndbox[3], label, difficult)
            else: #if shape is rotated box, save as rotated bounding box
                robndbox = LabelFile.convertPoints2RotatedBndBox(shape)
                writer.addRotatedBndBox(robndbox[0],robndbox[1],
                    robndbox[2],robndbox[3],robndbox[4],label,difficult)

        writer.save(targetFile=filename)

        # * add txt output
        filename_txt = os.path.join(os.path.dirname(filename), imgFileNameWithoutExt + '.txt')
        with open(filename_txt, 'w') as f:
            for shape in shapes:
                points = shape['points']
                label = shape['label']
                # Determine by flag if the coordinates are in order and in sequence.
                flag = True if ('_') in label else False
                if flag:
                    points = self.order_points(points, int(label.split('_')[1]))
                points = [int(p) for point in points for p in point]
                info = ''
                for point in points:
                    info += str(point) + ' '
                info += (label if not flag else label.split('_')[0]) + ' ' + ('1' if shape['difficult'] else '0') + '\n'
                f.write(info)
        
        # * add labelme json output

        return

    
    def order_points(self, pts, idx):
        pts = np.asarray(pts)
        #  根据x坐标对进行从小到大的排序
        sort_x = pts[np.argsort(pts[:, 0]), :]
        #  根据点x的坐标排序分别获取所有点中，位于最左侧和最右侧的点
        Left = sort_x[:2, :]
        Right = sort_x[2:, :]
        # 根据y坐标对左侧的坐标点进行从小到大排序，这样就能够获得左下角坐标点与左上角坐标点
        Left = Left[np.argsort(Left[:, 1])[::-1], :]
        # 根据y坐标对右侧的坐标点进行从小到大排序，这样就能够获得右上角坐标点与右下角坐标点
        Right = Right[np.argsort(Right[:, 1]), :]
        res = np.concatenate((Left, Right), axis=0)
        # 按选取的初始点进行拼接
        return np.concatenate((res[idx:], res[:idx]), axis=0).tolist() if idx < 4 else res.tolist()


    def toggleVerify(self):
        self.verified = not self.verified

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    # You Hao, 2017/06/121
    @staticmethod
    def convertPoints2RotatedBndBox(shape):
        points = shape['points']
        center = shape['center']
        direction = shape['direction']

        cx = center.x()
        cy = center.y()
        
        w = math.sqrt((points[0][0]-points[1][0]) ** 2 +
            (points[0][1]-points[1][1]) ** 2)

        h = math.sqrt((points[2][0]-points[1][0]) ** 2 +
            (points[2][1]-points[1][1]) ** 2)

        angle = direction % math.pi

        return (round(cx,4),round(cy,4),round(w,4),round(h,4),round(angle,6))
