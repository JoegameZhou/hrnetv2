# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train FasterRcnn(backbone:hrnetv2) and get checkpoint files."""

import os
import time
from datetime import datetime
from pprint import pprint
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn import SGD, Adam
from mindspore.common import set_seed
from mindspore.train.callback import SummaryCollector

from src.FasterRcnn.faster_rcnn import Faster_Rcnn
from src.model_utils.logger import get_logger
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr, multistep_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num

ms.context.set_context(max_call_depth=2000)

def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    print("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    print("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)
        files = os.listdir(config.data_path)
        for f in files:
            print(f)
        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        # ===========strat unzip==================================#
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
        # if not os.path.exists(sync_lock):
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)
        # print("Finish sync unzip data from {} to {}.".format(zip_file_1, save_dir_1))
        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
    print("=====================dataset file==============")
    for root,dirs,files in os.walk(config.data_path):
        for dir in dirs:
            print(os.path.join(root,dir))
        for file in files:
                print(os.path.join(root,file))
    print("===========pretained model file=================")
    for root,dirs,files in os.walk(config.load_path):
        for file in files:
                print(os.path.join(root,file))   
    config.save_checkpoint_path = config.output_path


def load_ckpt_to_network(net):
    load_path = config.pre_trained
    if config.enable_modelarts:
        print("pretain model name:{}".format(os.listdir(config.load_path)[0]))
        load_path = os.path.join(config.load_path,os.listdir(config.load_path)[0])
    if load_path != "":
        new_param = {}
        if config.finetune:
            param_not_load = ["learning_rate", "stat.rcnn.cls_scores.weight", "stat.rcnn.cls_scores.bias",
                              "stat.rcnn.reg_scores.weight", "stat.rcnn.reg_scores.bias",
                              "rcnn.cls_scores.weight", "rcnn.cls_scores.bias", "rcnn.reg_scores.weight",
                              "rcnn.reg_scores.bias", "accum.rcnn.cls_scores.weight", "accum.rcnn.cls_scores.bias",
                              "accum.rcnn.reg_scores.weight", "accum.rcnn.reg_scores.bias"
                              ]
            param_dict = ms.load_checkpoint(load_path, filter_prefix=param_not_load)
            for key, val in param_dict.items():
                # Correct previous misspellings
                key = key.replace("ncek", "neck")
                new_param[key] = val
        else:
            print(f"\n[{rank}]", "===> Loading from checkpoint:", load_path)
            param_dict = ms.load_checkpoint(load_path)
            key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                           'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                           'down_sample_layer.0.weight': 'conv_down_sample.weight',
                           'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                           'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                           }
            for oldkey in list(param_dict.keys()):
                if oldkey.startswith(('moment', "mean_square")):
                    # print(oldkey)
                    param_dict.pop(oldkey)
            for oldkey in list(param_dict.keys()):
                if not oldkey.startswith(
                        ('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                    data = param_dict.pop(oldkey)
                    newkey = 'backbone.' + oldkey
                    param_dict[newkey] = data
                    oldkey = newkey
                for k, v in key_mapping.items():
                    if k in oldkey:
                        newkey = oldkey.replace(k, v)
                        param_dict[newkey] = param_dict.pop(oldkey)
            for item in list(param_dict.keys()):
                if not item.startswith('backbone'):
                    param_dict.pop(item)
                    print("following keys is deleted")
                    print(item)

            for key, value in param_dict.items():
                tensor = value.asnumpy().astype(np.float32)
                param_dict[key] = Parameter(tensor, key)
            new_param = param_dict
            # print(new_param.keys())
        try:
            ms.load_param_into_net(net, new_param)
        except RuntimeError as ex:
            ex = str(ex)
            print("Traceback:\n", ex, flush=True)
            if "reg_scores.weight" in ex:
                exit("[ERROR] The loss calculation of faster_rcnn has been updated. "
                     "If the training is on an old version, please set `without_bg_loss` to False.")

    print(f"[{rank}]", "\tDone!\n")
    return net


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_fasterrcnn():
    """ train_fasterrcnn """
    print(f"\n[{rank}] - rank id of process")
    dataset_size, dataset = train_fasterrcnn_()

    print(f"\n[{rank}]", "===> Creating network...")
    net = Faster_Rcnn(config=config)
    net = net.set_train()
    net = load_ckpt_to_network(net)

    device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
    print(f"\n[{rank}]", "===> Device type:", device_type, "\n")
    if device_type == "Ascend":
        net.to_float(ms.float16)

    # 设置logger
    log_dir = config.log_dir
    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    logger = get_logger(log_dir, rank)
    logger.save_args(config)

    print(f"\n[{rank}]", "===> Creating loss, lr and opt objects...")
    loss = LossNet()
    if config.lr_type.lower() not in ("dynamic", "multistep"):
        raise ValueError("Optimize type should be 'dynamic' or 'dynamic'")
    if config.lr_type.lower() == "dynamic":
        lr = Tensor(dynamic_lr(config, dataset_size), ms.float32)
    else:
        lr = Tensor(multistep_lr(config, dataset_size), ms.float32)

    if config.opt_type.lower() not in ("sgd", "adam"):
        raise ValueError("Optimize type should be 'SGD' or 'Adam'")
    if config.opt_type.lower() == "sgd":
        opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                  weight_decay=config.weight_decay)
    else:
        opt = Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay)
    net_with_loss = WithLossCell(net, loss)
    print(f"[{rank}]", "\tDone!\n")
    net = TrainOneStepCell(net_with_loss, opt, scale_sense=config.loss_scale)
    print(f"\n[{rank}]", "===> Creating callbacks...")
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(logger, dataset_size, lr=lr.asnumpy(), per_print_times=20)
    cb = [time_cb, loss_cb]
    if config.log_summary:
        summary_collector = SummaryCollector(summary_dir)
        cb.append(summary_collector)
    print(f"[{rank}]", "\tDone!\n")

    print(f"\n[{rank}]", "===> Configurating checkpoint saving...")
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]
    print(f"[{rank}]", "\tDone!\n")

    if config.run_eval:
        from src.eval_callback import EvalCallBack
        from src.eval_utils import create_eval_mindrecord, apply_eval
        config.prefix = "FasterRcnn_eval.mindrecord"
        anno_json = os.path.join(config.data_path, "COCO2017/annotations/instances_val2017.json")  # if not on QiZhi: config.coco_root
        if hasattr(config, 'val_set'):
            anno_json = os.path.join(config.coco_root, config.val_set)
        # config.mindrecord_dir = os.path.join(config.coco_root, "FASTERRCNN_MINDRECORD")
        print(anno_json)
        mindrecord_path = os.path.join(config.mindrecord_dir, config.prefix)
        config.instance_set = "annotations/instances_val2017.json"

        # if not os.path.exists(mindrecord_path):
        #     config.mindrecord_file = mindrecord_path
        #     create_eval_mindrecord(config)
        eval_net = Faster_Rcnn(config)
        eval_cb = EvalCallBack(config, eval_net, apply_eval, dataset_size, mindrecord_path, anno_json,
                               save_checkpoint_path)
        cb += [eval_cb]
    print("creat mindrecord done!strat training!")
    model = Model(net)

    model.train(config.epoch_size, dataset, callbacks=cb)
    


if __name__ == '__main__':
    set_seed(1)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    # if not os.path.exists(config.mindrecord_dir):
    #         os.makedirs(config.mindrecord_dir)
    local_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    summary_dir = local_path + "/train/summary/"

    if config.device_target == "GPU":
        ms.set_context(enable_graph_kernel=False)
    if config.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        summary_dir += "thread_num_" + str(rank) + "/"
    else:
        rank = 0
        device_num = 1

    print()  # just for readability
    pprint(config)
    print(f"\n[{rank}] Please check the above information for the configurations\n\n", flush=True)

    train_fasterrcnn()
