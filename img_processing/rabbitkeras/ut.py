import os
import pydicom
import math
import shutil
import numpy as np
import cv2
from data_process import DataReader
from models.model import UnetModel
from matplotlib import pyplot as plt

bbox_data_dir = "data_rabbit_head_preprocess_reg_bbox"
mask_data_dir = "data_rabbit_head_preprocess_reg_mask"
dicom_data_dir = "dicom_data"
jpg_dir = "datasets/regnew"


def data_main():

    no_list = []
    lower_list = []
    full_list = []
    dicompng = 'dicompng'

    for f in sorted(os.listdir(dicompng)):
        f_dir = os.path.join(dicompng, f)
        print(f_dir)
        dirname = f_dir.split('/')[-1]
        print(dirname)
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

def copyxml():
    from shutil import copyfile

    for f in sorted(os.listdir(jpg_dir)):
        if os.path.isfile(f):
           continue
        f_dir = os.path.join(jpg_dir, f)
        print(f_dir)
        dirname = f_dir.split('/')[-1]
        print(dirname)
        xml_src = os.path.join(jpg_dir, 'full-contrast.xml')
        print(xml_src)
        xml_dst = os.path.join(f_dir, 'full-contrast.xml')
        print(xml_dst)
        copyfile(xml_src, xml_dst)


def rename():
    no_list = []
    lower_list = []
    full_list = []
    dir_list = []

    from shutil import copyfile
    alldir = 'dicom_all'
    if os.path.exists(alldir):
        shutil.rmtree(alldir)
    os.mkdir(alldir)

    for f in sorted(os.listdir(dicom_data_dir)):
        f_dir = os.path.join(dicom_data_dir, f)
        print(f_dir)
        dirname = f_dir.split('/')[-1]
        print(dirname)
        for img in os.listdir(f_dir):
            if "no" in img:
                no_src = os.path.join(f_dir, img)
                no_dst = os.path.join(alldir, img.replace(' ', ''))
                copyfile(no_src, no_dst)
            elif "low" in img:
                low_src = os.path.join(f_dir, img)
                low_dst = os.path.join(alldir, img.replace(' ', ''))
                copyfile(low_src, low_dst)
            elif "full" in img:
                full_src = os.path.join(f_dir, img)
                full_dst = os.path.join(alldir, img.replace(' ', ''))
                copyfile(full_src, full_dst)


def get_dicom():
    no_list = []
    lower_list = []
    full_list = []
    dir_list = []
    for f in sorted(os.listdir(dicom_data_dir)):
        f_dir = os.path.join(dicom_data_dir, f)
        print(f_dir)
        dirname = f_dir.split('/')[-1]
        print(dirname)
        dir_list.append(dirname)
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
#        print(target)
        target = float(target)
        return target

#    def get_tarnum(dictstr, tarstr):
#        #        wwind = dictstr.find('0028, 1051')
#        wwind = dictstr.find(tarstr)
#        if wwind is -1:
#            pass
#        wstart = dictstr.find('value=', wwind)
#        wwstart = dictstr.find('\'', wstart)
#        wwend = dictstr.find(',', wstart)
#        target = dictstr[wwstart + 1:wwend - 1]
#        print(target)
#        num = hex2dec(target)
#        print("num:", num)
##        target = float(target)
#        return target

    def hex2dec(string_num):
        import struct
        intn = struct.unpack('ii', string_num)
        print(intn)
        return str(int(string_num.upper(), 16))

    def get_tarimg(dcmdata):
        dcmdict = dcmdata.__dict__
        ww, wc = 0, 0
        for k, v in dcmdict.items():
#            print("k:{}, v:{}".format(k, v))
#            intercept = get_tarstr(dictstr, tarstr="0028, 1052")
#            slop= get_tarstr(dictstr, tarstr="0028, 1052")
            if isinstance(v, dict):
                dictstr = str(v)
#                print(dictstr)
                ww = get_tarstr(dictstr, tarstr='0028, 1051')
                wc = get_tarstr(dictstr, tarstr='0028, 1050')
#                width = get_tarnum(dictstr, tarstr='0028, 0010')
#                height = get_tarnum(dictstr, tarstr='0028, 0011')
#                print(wc, ww, width, height)

        if ww is 0 or wc is 0:
            print("illegal Window! FILE:{}, WW:{}, WC:{}".format(no_list[i], ww, wc))
            return None

        a = np.array([[0.5, 0.5], [-1, 1]])
        b = np.array([wc, ww])
        slover = np.linalg.solve(a, b)
        x, y = slover[0], slover[1]

        img = dcmdata.pixel_array
#        print("img max:{}, img min:{}, img mean:{}".format(img.argmax(), img.argmin(), img.mean()))

        img = np.where(img > y, 255, np.where(img < x, 0, (img - x) * 255 / ww))
#        print("{0:5.2f}".format(img.mean()))
        return img

    def predimg(model, img):
        input = np.reshape(img, (1, 240, 240, 1))
        res = model.predict(input)
        pred_mask = res[0]
        mask_img = pred_mask[:, :, 0] * 255
        mask_img = np.where(mask_img > 0, 255, 0)
        mask_img = mask_img.astype('uint8')
        mask = np.where(pred_mask > 0, 1, 0)
        mask = np.reshape(mask, (240, 240))
        pred_img = img * mask
        img1 = np.power(pred_img/255., 1/1.5) * 255
        img2 = np.power(pred_img/255., 1.5) * 255
        img1 = np.where(img1 > 255, 255, img1)
        img2 = np.where(img2 > 255, 255, img1)

        return mask_img, pred_img, img1, img2

#    unet = UnetModel()
#    model = unet.get_model()
#    filepath = "weights.118-0.03.hdf5"
#    model.load_weights(filepath)

    import pickle
    from tool.utils import mkdir_if_not_exist

    for i in range(len(no_list)):

        tardir = os.path.join(jpg_dir, dir_list[i])
        labeldict = pickle.load(open("datasets/patchlabel.p", "rb"))
        if dir_list[i] not in labeldict:
            continue
        x1, y1, x2, y2 = labeldict[dir_list[i]]

        label_pic_dict = pickle.load(open("datasets/brainlabel.p", "rb"))
        if dir_list[i] not in label_pic_dict:
            continue
        label_arr = label_pic_dict[dir_list[i]] / 255.

        mkdir_if_not_exist(tardir)

        try:
            dcm_full = pydicom.read_file(full_list[i])
#        print('dcm full file:', full_list[i])
            img_full = np.asarray(get_tarimg(dcm_full))
#        print(img_full.shape)
        except Exception as e:
            print('[Exception] Error {} file {} '.format(e, full_list[i]))
            exit(0)

        img_full_test = img_full.copy()
#        for bin in reversed(range(0, 2)):
#            cbin = 128 * bin
#            imgcb = np.where((img_full - cbin) >= 0, img_full_test, 0)
#            cv2.imwrite('{}/full-contrast-{}.png'.format(tardir, bin), imgcb[y1:y2, x1:x2])
#            img_full_test -= imgcb

        dcm_no = pydicom.read_file(no_list[i])
        print('dcm no file:', no_list[i])
        img_no = np.asarray(get_tarimg(dcm_no))
#        img_no = np.where((img_no-img_full) > 3, img_full, img_no)

#        img_no_test = img_no.copy()
#        for bin in reversed(range(0, 2)):
#            cbin = 128 * bin
#            imgcb = np.where((img_no - cbin) >= 0, img_no_test, 0)
#            cv2.imwrite('{}/no-contrast-{}.png'.format(tardir, bin), imgcb[y1:y2, x1:x2])
#            img_no_test -= imgcb

#        print("orig no image mean", img_orig_no.mean())
#        plt.hist(img_orig_no.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/orignohist_{}.png".format(tardir, i))
#        plt.close()
#
        dcm_low = pydicom.read_file(lower_list[i])
        print('dcm low file:', lower_list[i])
        img_low = np.asarray(get_tarimg(dcm_low))
#        img_low = np.where((img_low-img_full) > 3, img_full, img_low)

#        img_low_test = img_low.copy()
#        for bin in reversed(range(0, 2)):
#            cbin = 128 * bin
#            imgcb = np.where((img_low - cbin) >= 0, img_low_test, 0)
#            cv2.imwrite('{}/lower-contrast-{}.png'.format(tardir, bin), imgcb[y1:y2, x1:x2])
#            img_low_test -= imgcb

        img_full = img_full[y1:y2, x1:x2] #* label_arr
        img_no   = img_no[y1:y2, x1:x2] #* label_arr
        img_low  = img_low[y1:y2, x1:x2] #* label_arr
        img_no = np.where(img_no > img_full, img_full, img_no)
        img_low = np.where(img_low > img_full, img_full, img_low)
#        img_full_bar = np.where(img_full > (0.9*img_full.argmax()), 0, img_full)
#        full_roi_mean = img_full_bar.sum()/np.count_nonzero(img_full_bar)
#        print("full_roi_mean", full_roi_mean)
#
#        img_no_bar = np.where(img_no > (0.9*img_no.argmax()), 0, img_no)
#        no_roi_mean = img_no_bar.sum()/np.count_nonzero(img_no_bar)
#        print("no_roi_mean", no_roi_mean)
#
#        img_low_bar = np.where(img_low > (0.9*img_low.argmax()), 0, img_low)
#        low_roi_mean = img_low_bar.sum()/np.count_nonzero(img_low_bar)
#        print("low_roi_mean", low_roi_mean)
#
#        img_no = img_no * full_roi_mean / no_roi_mean
#        print("new img no roi mean", img_no.sum()/np.count_nonzero(img_no))
#        img_low = img_low * full_roi_mean /low_roi_mean
#        print("new img low roi mean", img_low.sum()/np.count_nonzero(img_low))

#        plt.hist(img_full.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/img_full_hist.png".format(tardir))
#        plt.close()
#
#        plt.hist(img_no.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/img_no_hist.png".format(tardir))
#        plt.close()
#
#        plt.hist(img_low.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/img_low_hist.png".format(tardir))
#        plt.close()

#        print("orig low image mean", img_orig_low.mean())
#        plt.hist(img_orig_low.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/origlowhist_{}.png".format(tardir, i))
#        plt.close()
#
#
#        img_mid_no = np.where(img_orig_no > img_orig_low, img_orig_low, img_orig_no)
#        print("middle no image mean", img_mid_no.mean())
#        plt.hist(img_mid_no.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/midnohist_{}.png".format(tardir, i))
#        plt.close()
#
#        img_low = np.where(img_orig_low > img_full, img_full, img_orig_low)
#        print("final low image mean", img_low.mean())
#
#        plt.hist(img_low.ravel(), 256, range=(1, 255))
#        img_no = np.where(img_mid_no > img_full, img_full, img_mid_no)
#        print("final no image mean", img_no.mean())
#
#        plt.hist(img_no.ravel(), 256, range=(1, 255))
#        plt.hist(img_full.ravel(), 256, range=(1, 255))
#        print("final full image mean", img_full.mean())
#        plt.show()
#
#        diff_low = img_low - img_no
#        plt.hist(diff_low.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/difflowhist_{}.png".format(tardir, i))
#        plt.close()
#        diff_full = img_full - img_low
#        plt.hist(diff_full.ravel(), 256, range=(1, 255))
#        plt.savefig("{}/difffullhist_{}.png".format(tardir, i))
#        plt.close()


#        img_no_mask, img_no_pred, _, _ = predimg(model, img_no)
#        img_low_mask, img_low_pred, _, _ = predimg(model, img_low)
#        img_full_mask, img_full_pred, gama1, gama2 = predimg(model, img_full)

#        ret, thresh = cv2.threshold(diff_full, 0, 255, cv2.THRESH_BINARY_INV)
#        kernel=np.ones((3,3),np.uint8)
#        opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
#        sure_bg = cv2.dilate(opening, kernel, iterations=3)

#        dist_transfrom=cv2.distanceTransform(opening,cv2.DIST_L2, 5)
        #cv2.imshow('dist_transfrom',dist_transfrom)
#        ret,sure_fg =cv2.threshold(dist_transfrom,0.7*dist_transfrom.max(),255,0)


#        blurred = cv2.GaussianBlur(diff_full, (9, 9), 0)
#        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
#        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
#        gradient = cv2.subtract(gradX, gradY)
#        gradient = cv2.convertScaleAbs(gradient)
#
#        newblurred = cv2.GaussianBlur(gradient, (9, 9), 0)
#        (_, thresh) = cv2.threshold(newblurred, 90, 255, cv2.THRESH_BINARY)



        cv2.imwrite('{}/no-contrast.png'.format(tardir), img_no)
        cv2.imwrite('{}/lower-contrast.png'.format(tardir), img_low)
        cv2.imwrite('{}/full-contrast.png'.format(tardir), img_full)
#        cv2.imwrite('{}/lower-diff.png'.format(tardir), diff_low)
#        cv2.imwrite('{}/full-diff.png'.format(tardir), diff_full)

#        cv2.imwrite('{}/opening.png'.format(tardir), opening)
#        cv2.imwrite('{}/bg.png'.format(tardir), sure_bg)
#        cv2.imwrite('{}/fg.png'.format(tardir), sure_fg)
#        cv2.imwrite('{}/gradient.png'.format(tardir), gradient)

#        cv2.imwrite('{}/img-no-mask.png'.format(tardir), img_no_mask)
#        cv2.imwrite('{}/imglowmask.png'.format(tardir), img_low_mask)
#        cv2.imwrite('{}/imgfullmask.png'.format(tardir), img_full_mask)
#        cv2.imwrite('{}/pred-no-contrast.jpg'.format(tardir), img_no_pred)
#        cv2.imwrite('{}/pred-lower-contrast.jpg'.format(tardir), img_low_pred)
#        cv2.imwrite('{}/pred-full-contrast.jpg'.format(tardir), img_full_pred)





if __name__ == '__main__':
#    rename()
    get_dicom()
#    copyxml()
#    data_main()


