import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from tool.utils import parse_voc_xml
import pickle

class DataReader():
    no_list, lower_list, full_list = [], [], []
    no_train, no_test, lower_train, lower_test, full_train, full_test = None, None, None, None, None, None
    no_data_list, lower_data_list, full_data_list = [], [], []

    def __init__(self, config):
        if config is None:
            from tool.config_utils import  process_config
            try:
                config = process_config('segmention_config.json')
                print(config)
            except Exception as e:
                print('[Exception] Config Error, %s' % e)
                exit(0)

        self.pic_data_dir = config.pic_data_dir
        self.dicom_data_dir = config.dicom_data_dir

        self.task = config.task
        self.ratio = 0.9
        self.suffix = config.suffix
        self.no_flag = config.no_flag
        self.low_flag = config.low_flag
        self.full_flag = config.full_flag
        self.label_flag = config.brain_label_flag
        self.mask_on = config.mask_on
        self.w = config.width
        self.h = config.height


    def get_picname(self):
        def islabel(img):
            if img.find('png') != -1:
                if img.find('pixels') != -1:
                    return True
            return False

        def is_no_img(imgname):
            return True if self.no_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        def is_low_img(imgname):
            return True if self.low_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        def is_full_img(imgname):
            return True if self.full_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        for f in os.listdir(self.pic_data_dir):
            f_dir = os.path.join(self.pic_data_dir, f)
            for img in sorted(os.listdir(f_dir)):
                if is_no_img(img):
                    self.no_list.append(os.path.join(f_dir, img))
                elif is_low_img(img):
                    self.lower_list.append(os.path.join(f_dir, img))
                elif is_full_img(img):
                    self.full_list.append(os.path.join(f_dir, img))

        print('nolist:', len(self.no_list))
        print('lowerlist:', len(self.lower_list))
        print('fulllist:', len(self.full_list))

    def gene_dataset(self):

        def get_img_inlist(img_list, i):
            from PIL import ImageFilter, ImageEnhance

            img = Image.open(img_list[i]).convert('L')
            img0 = img.transpose(Image.ROTATE_90)
            img1 = img.transpose(Image.ROTATE_180)
            img2 = img.transpose(Image.ROTATE_270)
            img3 = img.transpose(Image.FLIP_LEFT_RIGHT)
            img4 = img.transpose(Image.FLIP_TOP_BOTTOM)
            img5 = img.rotate(45)
            img6 = img0.rotate(45)
            img7 = img1.rotate(45)
            img8 = img2.rotate(45)
#            extend_list = [img, img0, img1, img2, img3, img4, img5, img6, img7, img8]
            extend_list = [img]


#            enhancer = ImageEnhance.Sharpness(img)
#
#            for i in range(8):
#                factor = i / 4.0
#
#            img1 = img.filter(ImageFilter.BLUR)
#            img1 = img.filter(ImageFilter.SHARPEN)
#            img1 = img.filter(ImageFilter.EDGE_ENHANCE)
#            img1 = img.filter(ImageFilter.FIND_EDGES)
#            img1 = img.filter(ImageFilter.BLUR)
            print("img {} name {}".format(i, img_list[i]))
            return extend_list

        for i in range(len(self.no_list)):
            no_extend_list = get_img_inlist(self.no_list, i)
            for noex in no_extend_list:
                self.no_data_list.append(np.array(noex))
            lower_extend_list = get_img_inlist(self.lower_list, i)
            for lowex in lower_extend_list:
                self.lower_data_list.append(np.array(lowex))
            full_extend_list = get_img_inlist(self.full_list, i)
            for fullex in full_extend_list:
                self.full_data_list.append(np.array(fullex))

        no_data = np.array(self.no_data_list)
        lower_data = np.array(self.lower_data_list)
        full_data = np.array(self.full_data_list)

        no_data = no_data.astype('float32') / 255.
        lower_data = lower_data.astype('float32') / 255.
        full_data = full_data.astype('float32') / 255.

        no_data = np.reshape(no_data, (len(no_data), self.w, self.h, 1))
        lower_data = np.reshape(lower_data, (len(lower_data), self.w, self.h, 1))
        full_data = np.reshape(full_data, (len(full_data), self.w, self.h, 1))

        # shuffle data
        np.random.seed(1)
        shuffle_indices = np.random.permutation(np.arange(len(full_data)))
#        print(type(shuffle_indices))
        no_shuffled = no_data[shuffle_indices]
#        print(type(no_shuffled))
        lower_shuffled = lower_data[shuffle_indices]
        full_shuffled = full_data[shuffle_indices]

        # partition data
        dev_sample_index = int(self.ratio * float(len(full_data)))
        self.no_train, self.no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
        self.lower_train, self.lower_test = lower_shuffled[:dev_sample_index], lower_shuffled[dev_sample_index:]
        self.full_train, self.full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]

    def init(self):
        self.get_picname()
        self.gene_dataset()

    def reg(self):
        self.get_picname()
        self.reg_picture()

    def get_show_data(self):
        return (self.no_data_list,
                self.lower_data_list,
                self.full_data_list)

    def get_aug_data(self):
        return (self.no_train,
                self.no_test,
                self.lower_train,
                self.lower_test,
                self.full_train,
                self.full_test)

class PatchDataReader(DataReader):
    bbox_list = []

    def __init__(self, config):
        super(PatchDataReader, self).__init__(config)
        self.w = config.width
        self.h = config.height
        self.bbox_flag = config.bbox_flag

    def hasbbox(self, path, labelfile):
        key = os.path.split(path)[-1]
        if key in labelfile:
            return True

#        for file in os.listdir(path):
#            if file.find('xml') != -1:
#                if file.find(self.bbox_flag) != -1:
#                    print("file : {} in {} has bbox".format(file, path))
        return False

    def get_picname_patch(self):

        def is_no_img(imgname):
            return True if self.no_flag in imgname and self.suffix in imgname else False

        def is_low_img(imgname):
            return True if self.low_flag in imgname and self.suffix in imgname else False

        def is_full_img(imgname):
            return True if self.full_flag in imgname and self.suffix in imgname else False


        label_dict = dict()
        labelfile = pickle.load(open("datasets/patchlabel.p", "rb"))
        for f in os.listdir(self.pic_data_dir):
            f_dir = os.path.join(self.pic_data_dir, f)
            key = os.path.split(f_dir)[-1]
            if self.hasbbox(f_dir, labelfile):
                bbox = labelfile[key]
                self.bbox_list.append(bbox)
                for img in sorted(os.listdir(f_dir)):
                    if is_no_img(img):
                        self.no_list.append(os.path.join(f_dir, img))
                    elif is_low_img(img):
                        self.lower_list.append(os.path.join(f_dir, img))
                    elif is_full_img(img):
                        self.full_list.append(os.path.join(f_dir, img))

        #                print("=========================={}:{}:{}:{}========================".format(len(self.no_list), len(self.lower_list), len(self.full_list), len(self.label_list)))
        #            else:
        #                print('not brain dir:', f_dir)

#        import pickle
#        pickle.dump(label_dict, open('datasets/patchlabel.p', "wb"))


        print('nolist:', len(self.no_list))
        print('lowerlist:', len(self.lower_list))
        print('fulllist:', len(self.full_list))
        print('labellist:', len(self.bbox_list))

    def gene_dataset(self):

        def get_img_inlist(img_list, i, bbox):
            img = Image.open(img_list[i]).convert('L')
            patch = img.crop(bbox)
            patch = np.array(patch).astype('float32')
            return patch

        from tool.utils import mkdir_if_not_exist
#        mkdir_if_not_exist('datasets/patch')

        for i in range(len(self.no_list)):
            #       for i in range(1):
            bbox_ = self.bbox_list[i]
            full_ = get_img_inlist(self.full_list, i, bbox_)
            lower_ = get_img_inlist(self.lower_list, i, bbox_)
#            lower_ = np.where(lower_ > full_, full_, lower_)
            no_ = get_img_inlist(self.no_list, i, bbox_)
#            no_ = np.where(no_ > full_, full_, no_)
#            mkdir_if_not_exist('datasets/patch/{}'.format(i))
#            cv2.imwrite('datasets/patch/{}/no-contrast.png'.format(i), no_)
#            cv2.imwrite('datasets/patch/{}/lower-contrast.png'.format(i), lower_)
#            cv2.imwrite('datasets/patch/{}/full-contrast.png'.format(i), full_)
            self.no_data_list.append(no_)
            self.lower_data_list.append(lower_)
            self.full_data_list.append(full_)

        no_data = np.array(self.no_data_list)
        lower_data = np.array(self.lower_data_list)
        full_data = np.array(self.full_data_list)
        print(no_data.shape)

        no_data = no_data.astype('float32') / 255.
        lower_data = lower_data.astype('float32') / 255.
        full_data = full_data.astype('float32') / 255.

        no_data = np.reshape(no_data, (len(no_data), self.w, self.h, 1))
        lower_data = np.reshape(lower_data, (len(lower_data), self.w, self.h, 1))
        full_data = np.reshape(full_data, (len(full_data), self.w, self.h, 1))

        # shuffle data
        np.random.seed(1)
        shuffle_indices = np.random.permutation(np.arange(len(full_data)))
        print(type(shuffle_indices))
        no_shuffled = no_data[shuffle_indices]
        print(type(no_shuffled))
        lower_shuffled = lower_data[shuffle_indices]
        full_shuffled = full_data[shuffle_indices]

        # partition data
        dev_sample_index = int(self.ratio * float(len(full_data)))
        self.no_train, self.no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
        self.lower_train, self.lower_test = lower_shuffled[:dev_sample_index], lower_shuffled[dev_sample_index:]
        self.full_train, self.full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]

    def init(self):
        self.get_picname_patch()
        self.gene_dataset()

    def get_aug_data(self):
        return (self.no_train,
                self.no_test,
                self.lower_train,
                self.lower_test,
                self.full_train,
                self.full_test)


class LabelDataReader(DataReader):
    label_list = []
    label_train, label_test = None, None
    full_label_train, full_label_test = None, None
    label_data_list, full_label_list = [], []
    label_name_dict = dict()
    label_pic_dict = dict()

    def __init__(self, config):
        super(LabelDataReader, self).__init__(config)

    def hasbrainlabel(self, path):
        for img in os.listdir(path):
            if img.find('pixels') != -1:
                return True
        return False


    def get_picname_brain(self):
        def islabel(img):
            if img.find('png') != -1:
                if img.find('pixels') != -1:
                    if img.find(self.label_flag) != -1:
                        return True
            return False

        def is_no_img(imgname):
            return True if self.no_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        def is_low_img(imgname):
            return True if self.low_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        def is_full_img(imgname):
            return True if self.full_flag in imgname and self.suffix in imgname and not islabel(imgname) else False

        for f in os.listdir(self.pic_data_dir):
            f_dir = os.path.join(self.pic_data_dir, f)
            if self.hasbrainlabel(f_dir):
#                print('===========================================================')
                key = os.path.split(f_dir)[-1]
                for img in sorted(os.listdir(f_dir)):
                    if islabel(img):
                        labeldir = os.path.join(f_dir, img)
                        self.label_list.append(labeldir)
                        self.label_name_dict[labeldir] = key
                    elif is_no_img(img):
                        self.no_list.append(os.path.join(f_dir, img))
                    elif is_low_img(img):
                        self.lower_list.append(os.path.join(f_dir, img))
                    elif is_full_img(img):
                        self.full_list.append(os.path.join(f_dir, img))
#                print("=========================={}:{}:{}:{}========================".format(len(self.no_list), len(self.lower_list), len(self.full_list), len(self.label_list)))
#            else:
#                print('not brain dir:', f_dir)

        print('nolist:', len(self.no_list))
        print('lowerlist:', len(self.lower_list))
        print('fulllist:', len(self.full_list))
        print('labellist:', len(self.label_list))

    ## TODO read vessel code
    def get_picname_vessel(self):

        for f in os.listdir(self.mask_data_dir):
            f_dir = os.path.join(self.mask_data_dir, f)
            if self.hasvessellabel(f_dir):
                for img in sorted(os.listdir(f_dir)):
                    if self.isvessellabel(img):
                        self.label_list.append(os.path.join(f_dir, img))
                    elif "no-contrast" in img and self.suffix in img:
                        self.no_list.append(os.path.join(f_dir, img))
                    elif "lower-contrast" in img and self.suffix in img:
                        self.lower_list.append(os.path.join(f_dir, img))
                    elif "full-contrast" in img and self.suffix in img:
                        self.full_list.append(os.path.join(f_dir, img))

        print('nolist:', len(self.no_list))
        print('lowerlist:', len(self.lower_list))
        print('fulllist:', len(self.full_list))
        print('labellist:', len(self.label_list))


#    def get_picname_dicom(self):
    def gene_dataset(self):

        def get_img_inlist(img_list, i):
            img = Image.open(img_list[i]).convert('L')
            img = np.array(img).astype('float32')
            return img

        for i in range(len(self.no_list)):
 #       for i in range(1):
            label_ = get_img_inlist(self.label_list, i)
            full_  = get_img_inlist(self.full_list, i)
            lower_ = get_img_inlist(self.lower_list, i)
            no_    = get_img_inlist(self.no_list, i)


            key = self.label_name_dict[self.label_list[i]]
            self.label_pic_dict[key] = label_

            self.label_data_list.append(label_)
            self.no_data_list.append(no_)
            self.lower_data_list.append(lower_)
            self.full_data_list.append(full_)
            self.full_label_list.append((label_/255.) * full_)

        no_data = np.array(self.no_data_list)
        lower_data = np.array(self.lower_data_list)
        full_data = np.array(self.full_data_list)
        label_data = np.array(self.label_data_list)
        full_label_data = np.array(self.full_label_list)

        print(no_data.shape)

        no_data = no_data.astype('float32') / 255.
        lower_data = lower_data.astype('float32') / 255.
        full_data = full_data.astype('float32') / 255.
        label_data = label_data.astype('float32') / 255.
        full_label_data = full_label_data.astype('float32') / 255.


        no_data = np.reshape(no_data, (len(no_data), self.w, self.h, 1))
        lower_data = np.reshape(lower_data, (len(lower_data), self.w, self.h, 1))
        full_data = np.reshape(full_data, (len(full_data), self.w, self.h, 1))
        label_data = np.reshape(label_data, (len(label_data), self.w, self.h, 1))
        full_label_data = np.reshape(full_label_data, (len(full_label_data), self.w, self.h, 1))

        # shuffle data
        np.random.seed(1)
        shuffle_indices = np.random.permutation(np.arange(len(full_data)))
        print(type(shuffle_indices))
        no_shuffled = no_data[shuffle_indices]
        print(type(no_shuffled))
        lower_shuffled = lower_data[shuffle_indices]
        full_shuffled = full_data[shuffle_indices]
        label_shuffled = label_data[shuffle_indices]
        full_label_shuffled = full_label_data[shuffle_indices]

        # partition data
        dev_sample_index = int(self.ratio * float(len(full_data)))
        self.no_train, self.no_test = no_shuffled[:dev_sample_index], no_shuffled[dev_sample_index:]
        self.lower_train, self.lower_test = lower_shuffled[:dev_sample_index], lower_shuffled[dev_sample_index:]
        self.full_train, self.full_test = full_shuffled[:dev_sample_index], full_shuffled[dev_sample_index:]
        self.label_train, self.label_test = label_shuffled[:dev_sample_index], label_shuffled[dev_sample_index:]
        self.full_label_train, self.full_label_test = full_label_shuffled[:dev_sample_index], label_shuffled[dev_sample_index:]

        import pickle
        pickle.dump(self.label_pic_dict, open('datasets/brainlabel.p', 'wb'))

    def init(self):
        self.get_picname_brain()
        self.gene_dataset()

    def get_seg_data(self):
        return (self.no_train,
                self.no_test,
                self.label_train,
                self.label_test)

    def get_aug_data(self):
        return (self.no_train,
                self.no_test,
                self.lower_train,
                self.lower_test,
                self.full_train,
                self.full_test,
                self.label_train,
                self.label_test,
                self.full_label_train,
                self.full_label_test)

    def get_show_data(self):
        return (self.no_data_list,
                self.lower_data_list,
                self.full_data_list,
                self.label_data_list)

#TODO add vessel dataset and read




if __name__ == '__main__':
    from tool.config_utils import process_config

    try:
        config = process_config('segmention_config.json')
    #    print(config)
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
#    dr = LabelDataReader(config)
#    dr.init()
#    no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test, full_label_train, full_label_test = dr.get_aug_data()
    dr = PatchDataReader(config)
    dr.init()
    no_train, no_test, lower_train, lower_test, full_train, full_test = dr.get_aug_data()
    print(no_train.shape)
    print(full_train.shape)



