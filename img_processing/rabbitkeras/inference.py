import os
from data_process import DataReader, LabelDataReader, PatchDataReader
# from infers.simple_mnist_infer import SimpleMnistInfer
from models.model import UnetModel
from tool.config_utils import process_config
import numpy as np
from tool.eval_utils import compute_psnr_mean, compute_ssim, compute_psnr_roi

if __name__ == '__main__':
    config = None

    try:
        config = process_config('segmention_config.json')
        print(config)
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    print('[INFO] Preparing Data...')
    datareader = None
    if config.task == 'aug':
        datareader = DataReader(config=config)
    elif config.task == 'aug_label':
        datareader = LabelDataReader(config=config)
    elif config.task == 'aug_patch':
        datareader = PatchDataReader(config=config)

    datareader.init()

    print('train aug task')
    no_train, no_test, lower_train, lower_test, full_train, full_test = datareader.get_aug_data()
    model = UnetModel(config=config).get_model()

    print('load model', model)
    model.load_weights('segAugBrain_best_weights.h5')
    res = model.predict([no_test, lower_test])

    listpsnr = list()
    listssim = list()
    opsnr = list()
    ossim = list()

    for i in range(len(res)):
        no_img = no_test[i]
        no_img = no_img[:, :, 0] * 255
        no_img = no_img.astype('uint8')

        lower_img = lower_test[i]
        lower_img = lower_img[:, :, 0] * 255
        lower_img = lower_img.astype('uint8')

        pred_img = res[i]
        pred_img = pred_img[:, :, 0] * 255
        pred_img = pred_img.astype('uint8')

        full_img = full_test[i]
        full_img = full_img[:, :, 0] * 255
        full_img = full_img.astype('uint8')

        listpsnr.append(compute_psnr_mean(pred_img, full_img))
        listssim.append(compute_ssim(pred_img, full_img))
        opsnr.append(compute_psnr_mean(lower_img, full_img))
        ossim.append(compute_ssim(lower_img, full_img))

    ndpsnr = np.asarray(listpsnr)
    ndssim = np.asarray(listssim)
    ndopsnr = np.asarray(opsnr)
    ndossim = np.asarray(ossim)

    print(ndopsnr.mean(), ndopsnr.std())
    print(ndossim.mean(), ndossim.std())

    print(ndpsnr.mean(), ndpsnr.std())
    print(ndssim.mean(), ndssim.std())

