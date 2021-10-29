main_train.py#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)



def rescale(fun):
    ''' 
        Smiled Monalisa  
    '''

#    image = plt.imread(os.path.join(sys.path[0], "monalisa.jpg"))
    image = cv2.imread("data_rabbit_head_preprocess_reg_bbox/5/reg-noframe-1-lower-contrast04.JPG", 0)
    height, width = image.shape
    print(width, height)

    arrange = 40
    drawgraid = 5
    piece = 3

    srcpoint = list()
    dstpoint = list()

    sampleNo = 100
    mu = np.array([[1, 5]])
    Sigma = np.array([[1, 0.5], [1.5, 3]])

#    from numpy.linalg import cholesky
#    R = cholesky(Sigma)
#    print(R)

    scale = 0.1

    for w in range(piece):
        for h in range(piece):
            x, y = int(w*(width-1)/piece), int(h*(height-1)/piece)
#            image[y, x] = (255, 0, 0)
            srcpoint.append((y, x))
            if x <= 20 or x >= 220 or y <= 20 or y >= 220:
                derx, dery = 0, 0
            else:
                derx, dery = int(np.random.randn()*scale*width/piece), int(np.random.randn()*scale*height/piece)
                print("derx, dery:", derx, dery)
            dstpoint.append((y+dery, x+derx))
            #print("{},{} -> {},{}".format(x, y, x+derx, y+dery))

    p = np.array(srcpoint)
    q = np.array(dstpoint)

#    p = np.array([
#        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225],
#        [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
#    ])
#    q = np.array([
#        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225],
#        [243, 225], [207, 238], [261, 237], [199, 253], [232, 281], [279, 249]
#    ])
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    #affine deformation
    #demo(mls_affine_deformation, mls_affine_deformation_inv, "Affine")
#    rescale(mls_affine_deformation_inv)
    rescale(mls_affine_deformation_inv)

    #similarity deformation
    #demo(mls_similarity_deformation, mls_similarity_deformation_inv, "Similarity")
    #demo2(mls_similarity_deformation_inv)
    rescale(mls_similarity_deformation_inv)

    #rigid deformation
    #demo(mls_rigid_deformation, mls_rigid_deformation_inv, "Rigid")
#    demo2(mls_rigid_deformation_inv)
    rescale(mls_rigid_deformation_inv)
