#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import shutil
from xml.dom import minidom
from xml.dom.minidom import parse
import cv2

def cpy_by_rename(base_path1, dst_txt_path):
    txt_list = os.listdir(base_path1)
    for txtf in txt_list:
        if txtf.endswith(".jpg"):
            src_name = txtf
            src_path1 = base_path1 + '/' + src_name
            src_img = cv2.imread(src_path1)
            h,w,c=src_img.shape
            
            #print(src_path2)
            dst_path= dst_txt_path + '/' + src_name[0:-3]+'xml'
            #src_txt_name = base_path + '/' + src_name
            doc = minidom.Document()
            booklist = doc.createElement("annotation")
            doc.appendChild(booklist)
            f = file(dst_path, 'w')
            doc.writexml(f)
            f.close()
            dst_domTree = parse(dst_path)
            #根元素
            annotation_node = dst_domTree.documentElement
          
            ##annotations = src_domTree.getElementsByTagName('annotation')
            ##annotation = annotations[0]
            #annotation_node = dst_domTree.createElement('annotation')

            folder_node = dst_domTree.createElement('folder')
            d_folder_v = dst_domTree.createTextNode('aaaa/bbbb/cccc')
            folder_node.appendChild(d_folder_v)
            annotation_node.appendChild(folder_node)

            filename_node = dst_domTree.createElement('filename')
            d_filename_v = dst_domTree.createTextNode('src_name')
            filename_node.appendChild(d_filename_v)
            annotation_node.appendChild(filename_node)


            src_v = 'Unkown'
            source_node = dst_domTree.createElement('source')
            d_width_v = dst_domTree.createTextNode(src_v)
            source_node.appendChild(d_width_v)
            annotation_node.appendChild(source_node)

            #annotation1 = source.getElementsByTagName('annotation')
            #annotation1_v = annotation1[0].childNodes[0].data
            #annotation1_node = dst_domTree.createElement('annotation')
            #d_annotation1_v = dst_domTree.createTextNode(annotation1_v)
            #annotation1_node.appendChild(d_annotation1_v)
            #source_node.appendChild(annotation1_node)

            ##size = annotation.getElementsByTagName('size')[0]              #size
            size_node = dst_domTree.createElement('size')
            annotation_node.appendChild(size_node)

            width_node = dst_domTree.createElement('width')
            d_width_v = dst_domTree.createTextNode(str(w))
            width_node.appendChild(d_width_v)
            size_node.appendChild(width_node)

            height_node = dst_domTree.createElement('height')
            d_height_v = dst_domTree.createTextNode(str(h))
            height_node.appendChild(d_height_v)
            size_node.appendChild(height_node)

            depth_node = dst_domTree.createElement('depth')
            d_depth_v = dst_domTree.createTextNode(str(c))
            depth_node.appendChild(d_depth_v)
            size_node.appendChild(depth_node)

           


            f = file(dst_path, 'wb')
            dst_domTree.writexml(f,addindent='    ', newl='\n')
            f.close()
            

if __name__ == '__main__':

    base_path1 = './empty_images'
    dst_txt_path = './empty_xml'

    cpy_by_rename(base_path1, dst_txt_path)
    '''
    src_path = '/home/wd/create_ocr_pics_to_lianghao/corpus/123'
    dst_path = '/home/wd/create_ocr_pics_to_lianghao/corpus/123/123.txt'
    split_to_oneLine(src_path, dst_path)
    '''
        



