"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10

"""
import os
from data_process import DataReader, LabelDataReader, PatchDataReader
# from infers.simple_mnist_infer import SimpleMnistInfer
from models.model import UnetModel
from trainers.segmention_trainer import SegmentionTrainer
from trainers.augmention_trainer import AugmentionTrainer
from trainers.augmention_label_trainer import AugmentionLabelTrainer
from tool.config_utils import process_config
import numpy as np


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] Reading Configs...')

    config = None

    try:
        config = process_config('segmention_config.json')
        print(config)
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
    # np.random.seed(47)  # 固定随机数

    print('[INFO] Preparing Data...')
    datareader = None
    if config.task == 'aug':
        datareader = DataReader(config=config)
    elif config.task == 'aug_label':
        datareader = LabelDataReader(config=config)
    elif config.task == 'aug_patch':
        datareader = PatchDataReader(config=config)

    datareader.init()

    if config.task == 'seg':
        no_train, no_test, label_train, label_test = datareader.get_seg_data()
        model = UnetModel(config=config).get_model()
        trainer = SegmentionTrainer(model=model,
                                    data=[no_train, no_test, label_train, label_test],
                                    config=config)
        trainer.train()
    elif config.task == 'aug' or config.task == 'aug_patch':
        print('train aug task')
        no_train, no_test, lower_train, lower_test, full_train, full_test = datareader.get_aug_data()
        model = UnetModel(config=config).get_model()
        print('train model', model)
        trainer = AugmentionTrainer(model=model,
                                    data=[no_train, no_test, lower_train, lower_test, full_train, full_test],
                                    config=config)
    elif config.task == 'aug_label':
        print('train aug label task')
        no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test, full_label_train, full_label_test = datareader.get_aug_data()
        model = UnetModel(config=config).get_model()
        print('train model', model)
        trainer = AugmentionLabelTrainer(model=model,
                                    data=[no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test, full_label_train, full_label_test],
                                    config=config)


    trainer.train()

    print('[INFO] Training...')
    print('[INFO] Finishing...')


if __name__ == '__main__':
    main_train()
    # test_main()
