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
"""FasterRcnn feature pyramid network."""
import mindspore
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)))


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = ms.common.initializer.initializer("XavierUniform", shape=shape, dtype=ms.float32).init_data()
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


class HrnetFpn(nn.Cell):
    """
    Hrfpn : proposed fpn for hrnet

    Args:
        in_channels (tuple) - Channel size of input feature maps.  //list
        out_channels (int) - Channel size output.
        num_outs (int) - Num of output features.

    Returns:
        Tuple, with tensors of same channel size.

    Examples:
        neck = FeatPyramidNeck([32,64,128,256], 256, 5)
        input_data = (normal(0,0.1,(1,c,1280//(4*2**i), 768//(4*2**i)),
                      dtype=np.float32) \
                      for i, c in enumerate(config.fpn_in_channels))
        x = neck(input_data)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs):
        super(HrnetFpn, self).__init__()
        self.num_outs = num_outs
        self.in_channels = in_channels
        self.reduction_conv = nn.layer.SequentialCell(
            [nn.Conv2d(in_channels=sum(in_channels),
                       out_channels=out_channels,
                       kernel_size=1, pad_mode='valid',
                       has_bias=True),
             ]
        )
        self.avgpool_op1 = ops.AvgPool(pad_mode="VALID", kernel_size=2**1, strides=2**1)
        self.avgpool_op2 = ops.AvgPool(pad_mode="VALID", kernel_size=2 ** 2, strides=2 ** 2)
        self.avgpool_op3 = ops.AvgPool(pad_mode="VALID", kernel_size=2 ** 3, strides=2 ** 3)
        self.avgpool_op4 = ops.AvgPool(pad_mode="VALID", kernel_size=2 ** 4, strides=2 ** 4)
        self.fpn_conv = nn.CellList()
        for i in range(5):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3, pad_mode='pad',
                padding=1
            ))

    def construct(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = [inputs[0]]
        for i in range(1, len(inputs)):
            outs.append(
                mindspore.ops.interpolate(inputs[i], scales=(1., 1., float(2 ** i), float(2 ** i)), mode="bilinear"))
        out = mindspore.ops.Concat(axis=1)(outs)
        out = self.reduction_conv(out)
        outs = [out, self.avgpool_op1(out), self.avgpool_op2(out), self.avgpool_op3(out), self.avgpool_op4(out)]
        outputs = ()
        for i in range(5):
            # tmp_out = self.fpn_conv[i](outs[i])
            outputs = outputs + (self.fpn_conv[i](outs[i]),)

        return outputs




if __name__ == '__main__':
    from hrnetv2_mindspore.HRNetW48_seg.src.config import hrnetw32_config as model_config

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    x = np.random.rand(2, 32, 272, 200)
    x = Tensor(input_data=x, dtype=mindspore.float32)
    x2 = np.random.rand(2, 64, 136, 100)
    x2 = Tensor(input_data=x2, dtype=mindspore.float32)
    x3 = np.random.rand(2, 128, 68, 50)
    x3 = Tensor(input_data=x3, dtype=mindspore.float32)
    x4 = np.random.rand(2, 256, 34, 25)
    x4 = Tensor(input_data=x4, dtype=mindspore.float32)
    input = [x, x2, x3, x4]
    network = FeatPyramidNeck(in_channels=[32, 64, 128, 256], out_channels=256, num_outs=5)
    # network = Setr()
    # dict = network.parameters_dict()
    # param_dict = load_checkpoint("vit_b_16_224.ckpt")
    # for i in dict.keys():
    #     print(i)
    # load_param_into_net(network, param_dict)
    a = network(input)
    print(a.shape)
