    #   
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

"""Evaluation for FasterRcnn"""
import os
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
import mindspore as ms
from mindspore.common import set_seed, Parameter

from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset, parse_json_annos_from_txt
from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.FasterRcnn.faster_rcnn import Faster_Rcnn
ms.context.set_context(max_call_depth=2000)


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

        print("Zip file path: ", zip_file_1)
        print("Unzip file save dir: ", save_dir_1)
        unzip(zip_file_1, save_dir_1)
        print("===Finish extract data synchronization===")
        print("Finish sync unzip data from {} to {}.".format(zip_file_1, save_dir_1))
        config.checkpoint_path = config.output_path

@moxing_wrapper(pre_process=modelarts_pre_process)
def fasterrcnn_eval(dataset_path, ckpt_path, anno_path):
    """FasterRcnn evaluation."""
    if config.enable_modelarts:
        print("ckpt:")
        # ckpt_path = os.path.join(config.load_path, config.load_path[0])
        ckpt_path = "/cache/checkpoint_path/hrnet.ckpt"
        print(ckpt_path)
        anno_path = config.data_path + "/COCO2017/annotations/instances_val2017.json"
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
    ds = create_fasterrcnn_dataset(config, dataset_path, batch_size=config.test_batch_size, is_training=False)
    net = Faster_Rcnn(config)

    try:
        param_dict = ms.load_checkpoint(ckpt_path)
    except RuntimeError as ex:
        ex = str(ex)
        print("Traceback:\n", ex, flush=True)
        if "reg_scores.weight" in ex:
            exit("[ERROR] The loss calculation of faster_rcnn has been updated. "
                 "If the training is on an old version, please set `without_bg_loss` to False.")

    # in previous version of code there was a typo in layer name 'fpn_neck': it was 'fpn_ncek'
    # in order to make backward compatibility with checkpoints created with that typo
    # we need to manually check and rename that layer in param_dict
    for key, value in param_dict.items():
        if key.startswith('fpn_ncek'):
            new_key = key.replace('fpn_ncek', 'fpn_neck')
            param_dict[new_key] = param_dict.pop(key)
            print(f"param_dict fixed typo: {key} renamed to {new_key}")

    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    print(param_dict.keys())
    ms.load_param_into_net(net, param_dict)

    net.set_train(False)
    device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(ms.float16)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []

    if config.dataset != "coco":
        dataset_coco = COCO()
        dataset_coco.dataset, dataset_coco.anns, dataset_coco.cats, dataset_coco.imgs = dict(), dict(), dict(), dict()
        dataset_coco.imgToAnns, dataset_coco.catToImgs = defaultdict(list), defaultdict(list)
        dataset_coco.dataset = parse_json_annos_from_txt(anno_path, config)
        dataset_coco.createIndex()
    else:
        dataset_coco = COCO(anno_path)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = config.num_gts
    for data in ds.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        start = time.time()
        # run net
        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
            outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")

    coco_eval(result_files, eval_types, dataset_coco,
              single_result=True)
    print("\nEvaluation done!")


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_fasterrcnn():
    """ eval_fasterrcnn """
    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    start_time = time.time()
    fasterrcnn_eval(mindrecord_file, config.checkpoint_path, config.anno_path)
    end_time = time.time()
    total_time = end_time - start_time
    print("\nDone!\nTime taken: {:.2f} seconds".format(total_time))

    # config.eval_result_path = os.path.abspath("./eval_result")



if __name__ == '__main__':
    set_seed(1)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    eval_fasterrcnn()
