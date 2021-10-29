


import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import cv2



def get_evaluate():
    no_list = []
    lower_list = []
    full_list = []
    for f in os.listdir(bbox_data_dir):
        f_dir = os.path.join(bbox_data_dir, f)
        for img in os.listdir(f_dir):
            if "no-contrast" in img:
                no_list.append(os.path.join(f_dir, img))
            elif "lower-contrast" in img:
                lower_list.append(os.path.join(f_dir, img))
            elif "full-contrast" in img:
                full_list.append(os.path.join(f_dir, img))
    print(len(no_list))
    print(len(lower_list))
    print(len(full_list))

    no_data_list = []
    lower_data_list = []
    full_data_list = []

    for i in range(len(no_list)):
        img_no = np.array(Image.open(no_list[i]).convert('L'))
        no_data_list.append(img_no)
        img_lower = np.array(Image.open(lower_list[i]).convert('L'))
        lower_data_list.append(img_lower)
        img_full = np.array(Image.open(full_list[i]).convert('L'))
        full_data_list.append(img_full)
    no_data = np.array(no_data_list)
    lower_data = np.array(lower_data_list)
    full_data = np.array(full_data_list)

    no_data = no_data.astype('float32') / 255.
    lower_data = lower_data.astype('float32') / 255.
    full_data = full_data.astype('float32') / 255.
    print(no_data.shape)
    no_data = np.reshape(no_data, (len(no_data), 240, 240, 1))
    lower_data = np.reshape(lower_data, (len(lower_data), 240, 240, 1))
    full_data = np.reshape(full_data, (len(full_data), 240, 240, 1))
    print(no_data.shape)

    return no_data, lower_data, full_data


def parse_voc_xml(xml_file):
    texts = []

    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    path = tree.find('path').text

    # imgname = os.path.basename(path)
    imgpath = os.path.normpath(path).split('/')
    if len(imgpath) <= 1:
        imgpath = os.path.normpath(path).split('\\')
    imgname = imgpath[-1]

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        sx1 = int(bbox.find('xmin').text)
        sy1 = int(bbox.find('ymin').text)
        sx2 = int(bbox.find('xmax').text)
        sy2 = int(bbox.find('ymax').text)

    w = sx2 - sx1
    h = sy2 - sy1
    text = (sx1, sy1, sx2, sy2)

    return text


def data_aug(shape=(240, 240)):
    w, h = shape
    no_list = []
    lower_list = []
    full_list = []
    label_list = []
    for f in os.listdir(bbox_data_dir):
        f_dir = os.path.join(bbox_data_dir, f)
        if len(os.listdir(f_dir)) == 3:
            continue
        elif len(os.listdir(f_dir)) == 4:
            for img in sorted(os.listdir(f_dir)):
                if "xml" in img:
                    label_list.append(parse_voc_xml(os.path.join(f_dir, img)))
                elif "no-contrast" in img:
                    no_list.append(os.path.join(f_dir, img))
                elif "lower-contrast" in img:
                    lower_list.append(os.path.join(f_dir, img))
                elif "full-contrast" in img:
                    full_list.append(os.path.join(f_dir, img))

    print(len(no_list))
    print(len(lower_list))
    print(len(full_list))
    print(len(label_list))

    no_data_list = []
    lower_data_list = []
    full_data_list = []

    def get_img_inlist(img_list, i):
        img = Image.open(img_list[i]).convert('L')
        img = np.array(img)
        return img

    for i in range(len(no_list)):
        no_data_list.append(get_img_inlist(no_list, i))
        lower_data_list.append(get_img_inlist(lower_list, i))
        full_data_list.append(get_img_inlist(full_list, i))

    no_data = np.array(no_data_list)
    lower_data = np.array(lower_data_list)
    full_data = np.array(full_data_list)

    return no_data, lower_data, full_data


def claheaug(img, limit):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(2, 2))
    imgc = clahe.apply(img)
    return imgc


def get_pic_bbox(shape, aug=None):
    if shape is not None:
        w, h = shape
    else:
        w, h = (240, 240)
    no_list = []
    lower_list = []
    full_list = []
    label_data_list = []
    for f in os.listdir(bbox_data_dir):
        f_dir = os.path.join(bbox_data_dir, f)
        if len(os.listdir(
                f_dir)) == 3:  ##FIXME bug need fix, now is not work, should not count file number, need find xml
            continue
        elif len(os.listdir(f_dir)) == 4:
            for img in sorted(os.listdir(f_dir)):
                if "xml" in img:
                    x1, y1, x2, y2 = parse_voc_xml(os.path.join(f_dir, img))
                    bbox_pic = np.ones((w, h)) * 0.1
                    bbox_pic[y1:y2, x1:x2] = 1
                    #                    Image.fromarray(bbox_pic).convert('L').save('bbox.png')
                    label_data_list.append(bbox_pic)
                elif "no-contrast" in img:
                    no_list.append(os.path.join(f_dir, img))
                elif "lower-contrast" in img:
                    lower_list.append(os.path.join(f_dir, img))
                elif "full-contrast" in img:
                    full_list.append(os.path.join(f_dir, img))

    print(len(no_list))
    print(len(lower_list))
    print(len(full_list))
    print('labellist:', len(label_data_list))

    no_data_list = []
    lower_data_list = []
    full_data_list = []

    def get_img_inlist(img_list, i):
        img = Image.open(img_list[i]).convert('L')
        img = np.array(img)
        if aug is not None:
            img = aug(img, limit=1)
        return img

    for i in range(len(no_list)):
        no_data_list.append(get_img_inlist(no_list, i))
        lower_data_list.append(get_img_inlist(lower_list, i))
        full_data_list.append(get_img_inlist(full_list, i))

    return no_data_list, lower_data_list, full_data_list, label_data_list


def get_orig_pic(shape, aug=None):
    if shape is not None:
        w, h = shape
    else:
        w, h = (240, 240)
    no_list = []
    lower_list = []
    full_list = []
    for f in os.listdir(bbox_data_dir):
        f_dir = os.path.join(bbox_data_dir, f)
        for img in sorted(os.listdir(f_dir)):
            if img.find('JPG') == -1: continue
            if "no-contrast" in img:
                no_list.append(os.path.join(f_dir, img))
            elif "lower-contrast" in img:
                lower_list.append(os.path.join(f_dir, img))
            elif "full-contrast" in img:
                full_list.append(os.path.join(f_dir, img))

    print('nolist:', len(no_list))
    print('lowerlist:', len(lower_list))
    print('fulllist:', len(full_list))

    no_data_list = []
    lower_data_list = []
    full_data_list = []

    def get_img_inlist(img_list, i):
        img = Image.open(img_list[i]).convert('L')
        img = np.array(img)
        if aug is not None:
            img = aug(img, limit=1)
        return img

    for i in range(len(no_list)):
        no_data_list.append(get_img_inlist(no_list, i))
        lower_data_list.append(get_img_inlist(lower_list, i))
        full_data_list.append(get_img_inlist(full_list, i))

    return no_data_list, lower_data_list, full_data_list


def get_label_set(shape=(240, 240), label=False):
    #        no_data_list, lower_data_list, full_data_list = get_labelcut_pic(shape, aug) if label is True else get_orig_pic(shape, aug)
    #         no_data_list, lower_data_list, full_data_list = get_orig_pic(shape, aug)

    w, h = shape
    aug = claheaug
    no_data_list, lower_data_list, full_data_list, label_data_list = get_pic_label(shape, aug=None)

    no_data = np.array(no_data_list)
    lower_data = np.array(lower_data_list)
    full_data = np.array(full_data_list)
    label_data = np.array(label_data_list)

    no_data = no_data.astype('float32') / 255.
    lower_data = lower_data.astype('float32') / 255.
    full_data = full_data.astype('float32') / 255.
    label_data = label_data.astype('float32') / 255.
    print(no_data.shape)
    no_data = np.reshape(no_data, (len(no_data), w, h, 1))
    lower_data = np.reshape(lower_data, (len(lower_data), w, h, 1))
    full_data = np.reshape(full_data, (len(full_data), w, h, 1))
    label_data = np.reshape(label_data, (len(full_data), w, h, 1))
    print(no_data.shape)

    # shuffle data
    np.random.seed(1)
    shuffle_indices = np.random.permutation(np.arange(len(full_data)))
    print(type(shuffle_indices))
    no_shuffled = no_data[shuffle_indices]
    print(type(no_shuffled))
    lower_shuffled = lower_data[shuffle_indices]
    full_shuffled = full_data[shuffle_indices]
    label_shuffled = label_data[shuffle_indices]

    # partition data
    dev_sample_index = int(0.9 * float(len(full_data)))
    no_train, no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
    lower_train, lower_test = lower_shuffled[:dev_sample_index], lower_shuffled[dev_sample_index:]
    full_train, full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]
    label_train, label_test = label_shuffled[:dev_sample_index], label_shuffled[dev_sample_index:]

    return no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test


def get_bbox_set(shape=(240, 240)):
    w, h = shape
    aug = claheaug
    no_data_list, lower_data_list, full_data_list, label_data_list = get_pic_bbox(shape, aug=None)

    no_data = np.array(no_data_list)
    lower_data = np.array(lower_data_list)
    full_data = np.array(full_data_list)
    label_data = np.array(label_data_list)

    no_data = no_data.astype('float32') / 255.
    lower_data = lower_data.astype('float32') / 255.
    full_data = full_data.astype('float32') / 255.
    print(label_data.shape)
    no_data = np.reshape(no_data, (len(no_data), w, h, 1))
    lower_data = np.reshape(lower_data, (len(lower_data), w, h, 1))
    full_data = np.reshape(full_data, (len(full_data), w, h, 1))
    label_data = np.reshape(label_data, (len(full_data), w, h, 1))

    # shuffle data
    np.random.seed(1)
    shuffle_indices = np.random.permutation(np.arange(len(full_data)))
    print(type(shuffle_indices))
    no_shuffled = no_data[shuffle_indices]
    print(type(no_shuffled))
    lower_shuffled = lower_data[shuffle_indices]
    full_shuffled = full_data[shuffle_indices]
    label_shuffled = label_data[shuffle_indices]

    # partition data
    dev_sample_index = int(0.9 * float(len(full_data)))
    no_train, no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
    lower_train, lower_test = lower_shuffled[:dev_sample_index], lower_shuffled[dev_sample_index:]
    full_train, full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]
    label_train, label_test = label_shuffled[:dev_sample_index], label_shuffled[dev_sample_index:]

    return no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test


def get_dicom():
    no_list = []
    lower_list = []
    full_list = []
    for f in os.listdir(dicom_data_dir):
        f_dir = os.path.join(dicom_data_dir, f)
        print(f_dir)
        for img in os.listdir(f_dir):
            if "no" in img:
                no_list.append(os.path.join(f_dir, img))
            elif "low" in img:
                lower_list.append(os.path.join(f_dir, img))
            elif "full" in img:
                full_list.append(os.path.join(f_dir, img))
    print(len(no_list))
    print(len(lower_list))
    print(len(full_list))

    no_data_list = []
    merge_data_list = []
    full_data_list = []

    import pydicom
    import math
    import shutil

    if os.path.exists(jpg_dir):
        shutil.rmtree(jpg_dir)
    os.mkdir(jpg_dir)

    def get_tarstr(dictstr, tarstr):
        #        wwind = dictstr.find('0028, 1051')
        wwind = dictstr.find(tarstr)
        if wwind is -1:
            pass
        wstart = dictstr.find('value=', wwind)
        wwstart = dictstr.find('\'', wstart)
        wwend = dictstr.find(',', wstart)
        target = dictstr[wwstart + 1:wwend - 1]
        target = float(target)
        return target

    def get_tarimg(dcmdata):
        dcmdict = dcmdata.__dict__
        ww, wc = 0, 0
        for k, v in dcmdict.items():
            if isinstance(v, dict):
                dictstr = str(v)
                # print(dictstr)
                ww = get_tarstr(dictstr, tarstr='0028, 1051')
                wc = get_tarstr(dictstr, tarstr='0028, 1050')
                print(wc, ww)

        if ww is 0 or wc is 0:
            print("illegal Window! FILE:{}, WW:{}, WC:{}".format(no_list[i], ww, wc))
            return None

        a = np.array([[0.5, 0.5], [-1, 1]])
        b = np.array([wc, ww])
        slover = np.linalg.solve(a, b)
        x, y = slover[0], slover[1]

        img = dcmdata.pixel_array

        img = np.where(img > y, 255, np.where(img < x, 0, (img - x) * 255 / ww))
        print("{0:5.2f}".format(img.mean()))
        return img

    for i in range(len(no_list)):
        #        if i != 0:
        #            continue
        dcm_no = pydicom.read_file(no_list[i])
        dcm_low = pydicom.read_file(lower_list[i])
        dcm_full = pydicom.read_file(full_list[i])

        img_no = get_tarimg(dcm_no)
        img_low = get_tarimg(dcm_low)
        img_full = get_tarimg(dcm_full)

        img_no = np.reshape(img_no, (240, 240, 1))
        img_low = np.reshape(img_low, (240, 240, 1))
        img_merge = np.concatenate((img_no, img_low), axis=2)

        print(img_merge.shape)

        no_data_list.append(img_no)
        merge_data_list.append(img_merge)
        full_data_list.append(img_full)

    no_data = np.array(no_data_list)
    merge_data = np.array(merge_data_list)
    full_data = np.array(full_data_list)

    no_data = no_data.astype('float32') / 255.
    merge_data = merge_data.astype('float32') / 255.
    full_data = full_data.astype('float32') / 255.

    no_data = np.reshape(no_data, (len(no_data), 240, 240, 1))
    merge_data = np.reshape(merge_data, (len(no_data), 240, 240, 2))
    full_data = np.reshape(full_data, (len(full_data), 240, 240, 1))
    print(no_data.shape)

    # shuffle data
    np.random.seed(1)
    shuffle_indices = np.random.permutation(np.arange(len(full_data)))
    no_shuffled = no_data[shuffle_indices]
    merge_shuffled = merge_data[shuffle_indices]
    full_shuffled = full_data[shuffle_indices]

    # partition data
    dev_sample_index = int(0.9 * float(len(full_data)))
    no_train, no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
    merge_train, merge_test = merge_shuffled[:dev_sample_index], merge_shuffled[dev_sample_index:]
    full_train, full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]

    return no_train, no_test, merge_train, merge_test, full_train, full_test