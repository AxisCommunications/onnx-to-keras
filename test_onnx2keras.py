import warnings
from io import BytesIO
from tempfile import NamedTemporaryFile

import onnx
import torch.nn
from torch.nn import Module
import torch.nn.functional as F
from torchvision import models
from numpy.testing import assert_almost_equal
import numpy as np
import tensorflow as tf

from onnx2keras import onnx2keras, compatible_data_format, OnnxConstant, OnnxTensor, InterleavedImageBatch, \
    ensure_data_format, OptimizationMissingWarning


def make_onnx_model(net, indata):
    fd = BytesIO()
    torch.onnx.export(net, indata, fd)
    fd.seek(0)
    return onnx.load(fd)


def convert_and_compare_output(net, indata, precition=5, image_out=True, savable=True, missing_optimizations=False):
    torch_indata = torch.tensor(indata)
    y1 = net(torch_indata).detach().numpy()
    onnx_model = make_onnx_model(net, torch.zeros_like(torch_indata))
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        kernas_net = onnx2keras(onnx_model)
        warns = [w for w in warns if w.category is OptimizationMissingWarning]
        if not missing_optimizations:
            assert len(warns) == 0
    if savable:
        with NamedTemporaryFile() as f:
            f.close()
            kernas_net.save(f.name)
    y2 = kernas_net.predict(indata.transpose(0, 2, 3, 1))
    if image_out:
        y2 = y2.transpose(0, 3, 1, 2)
    assert_almost_equal(y1, y2, precition)
    return kernas_net

class GlobalAvgPool(Module):
    def forward(self, x):
        return x.mean([2, 3])


class TestUtils:
    def test_compatible_data_format(self):
        assert compatible_data_format(OnnxConstant, OnnxConstant)
        assert compatible_data_format(OnnxTensor, OnnxTensor)
        assert compatible_data_format(OnnxConstant, OnnxTensor)
        assert compatible_data_format(OnnxTensor, OnnxConstant)
        assert compatible_data_format(InterleavedImageBatch, InterleavedImageBatch)
        assert not compatible_data_format(OnnxTensor, InterleavedImageBatch)
        assert not compatible_data_format(OnnxConstant, InterleavedImageBatch)
        assert not compatible_data_format(InterleavedImageBatch, OnnxTensor)
        assert not compatible_data_format(InterleavedImageBatch, OnnxConstant)


class TestOnnx:
    def test_conv(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 7), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_no_bias(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 7, bias=False), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_padding(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU())
        x = np.random.rand(1, 1, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_prelu(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 7), torch.nn.PReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_prelu_per_channel(self):
        act = torch.nn.PReLU(num_parameters=16)
        act.weight[:] = torch.tensor(range(16))
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 7), act)
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x, 5)

    def test_maxpool(self):
        net = torch.nn.Sequential(torch.nn.MaxPool2d(2))
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_maxpool_resnet(self):
        net = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = np.random.rand(1, 192, 272, 64).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_concat(self):
        for axis in range(1,4):
            class Dbl(torch.nn.Module):
                def forward(self, x):
                    return torch.cat((x, x), axis)
            x = np.random.rand(1, 3, 224, 224).astype(np.float32)
            convert_and_compare_output(Dbl(), x)

    def test_conv_transpose(self):
        net = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 5, 2), torch.nn.ReLU())
        x = np.random.rand(1, 3, 112, 112).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_transpose_padding(self):
        net = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 4, 2, padding=1), torch.nn.ReLU())
        x = np.random.rand(1, 3, 112, 112).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_different_padding(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=(3, 4)))
        x = np.random.rand(1, 3, 384, 544).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_transpose_no_bias(self):
        net = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 5, 2, bias=False), torch.nn.ReLU())
        x = np.random.rand(1, 3, 112, 112).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_stride2_padding_strange(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        x = np.random.rand(1, 3, 384, 544).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_conv_stride2_padding_simple_odd(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1))
        x = np.random.rand(1, 3, 223, 223).astype(np.float32)
        kernas_net = convert_and_compare_output(net, x)
        assert [l.__class__.__name__ for l in kernas_net.layers] == ['InputLayer', 'Conv2D']

    def test_conv_stride2_padding_simple_even(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1))
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        kernas_net = convert_and_compare_output(net, x)
        # assert [l.__class__.__name__ for l in kernas_net.layers] == ['InputLayer', 'Conv2D']

    def test_batchnorm(self):
        bn = torch.nn.BatchNorm2d(3)
        bn.running_mean.uniform_()
        bn.running_var.uniform_()
        net = torch.nn.Sequential(bn, torch.nn.ReLU())
        net.eval()
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_clamp(self):
        class Clamp(Module):
            def forward(self, x):
                return torch.clamp(x, 0.3, 0.7)
        net = torch.nn.Sequential(torch.nn.ReLU(), Clamp(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x, savable=False)

    def test_relu6(self):
        class Clamp(Module):
            def forward(self, x):
                return torch.clamp(x, 0, 6)
        net = torch.nn.Sequential(torch.nn.ReLU(), Clamp(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_depthwise(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 7, groups=3), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_depthwise_no_bias(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 7, groups=3, bias=False), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_add(self):
        class AddTst(Module):
            def __init__(self):
                Module.__init__(self)
                self.conv1 = torch.nn.Conv2d(3, 3, 7)
                self.conv2 = torch.nn.Conv2d(3, 3, 7)
            def forward(self, x):
                return self.conv1(x).relu_() + self.conv2(x).relu_()
        net = torch.nn.Sequential(AddTst(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_global_avrage_pooling(self):
        net = torch.nn.Sequential(GlobalAvgPool(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_dropout(self):
        net = torch.nn.Sequential(GlobalAvgPool(), torch.nn.Dropout(), torch.nn.ReLU())
        net.eval()
        x = np.random.rand(1, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_linear(self):
        net = torch.nn.Sequential(GlobalAvgPool(), torch.nn.Linear(3, 8), torch.nn.ReLU())
        net.eval()
        x = np.random.rand(5, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_linear_no_bias(self):
        net = torch.nn.Sequential(GlobalAvgPool(), torch.nn.Linear(3, 8, bias=False), torch.nn.ReLU())
        net.eval()
        x = np.random.rand(5, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_mobilenet_v2(self):
        net = models.mobilenet_v2()
        net.eval()
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_avg_pool_pad(self):
        class PadTst(Module):
            def forward(self, x):
                return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        net = torch.nn.Sequential(PadTst(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_avg_pool_pad_asym(self):
        class PadTst(Module):
            def forward(self, x):
                return F.avg_pool2d(x, kernel_size=(3, 6), stride=(1, 2), padding=(1, 2))
        net = torch.nn.Sequential(PadTst(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_gloabl_avg_pool(self):
        class AvgTst(Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))
        net = torch.nn.Sequential(AvgTst(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_flatten(self):
        class Tst(Module):
            def forward(self, x):
                return torch.flatten(x, 1)
        net = torch.nn.Sequential(Tst(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 1, 1).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_vector_pad(self):
        class VectorPad2D(Module):
            def forward(self, x):
                tt = [torch.nn.functional.pad(x[:, i:i + 1], [1,1,1,1], 'constant', [1,2,3][i])
                      for i in range(x.shape[1])]
                return torch.cat(tt, 1)
        net = torch.nn.Sequential(VectorPad2D(), torch.nn.ReLU())
        x = np.random.rand(2, 3, 5, 5).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_vector_pad_addhack(self):
        class VectorPad2D(Module):
            def forward(self, x):
                c = torch.tensor([1,2,3]).reshape(1, 3, 1, 1)
                return torch.nn.functional.pad(x - c, [1,1,1,1]) + c
        net = torch.nn.Sequential(VectorPad2D(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 5, 5).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_vector_pad_addhack_asym(self):
        class VectorPad2D(Module):
            def forward(self, x):
                c = torch.tensor([1,2,3]).reshape(1, 3, 1, 1)
                return torch.nn.functional.pad(x - c, [1,0,1,0]) + c
        net = torch.nn.Sequential(VectorPad2D(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 5, 5).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_sigmoid(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 7), torch.nn.Sigmoid())
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_upsample_nearest(self):
        net = torch.nn.Sequential(torch.nn.UpsamplingNearest2d(scale_factor=2), torch.nn.ReLU())
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_eq_mul(self):
        class EqProd(Module):
            def forward(self, x):
                maxmap = F.max_pool2d(x, 3, 1, 1, 1, False, False)
                return x * (maxmap == x)
        net = torch.nn.Sequential(EqProd(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 5, 5).astype(np.float32)
        convert_and_compare_output(net, x)

    def test_adaptive_avgpool_reshape(self):
        class Net(Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        net = torch.nn.Sequential(Net(), torch.nn.ReLU())
        x = np.random.rand(1, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)
        x = np.random.rand(4, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_bmm(self):
        class Net(Module):
            def forward(self, x):
                x = x.reshape(1, 16, 16)
                return torch.bmm(x, x)
        net = torch.nn.Sequential(Net(), torch.nn.ReLU())
        x = np.random.rand(1, 1, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_matmul(self):
        class Net(Module):
            def forward(self, x):
                x = x.reshape(1, 1, 16, 16)
                return torch.matmul(x, x)
        net = torch.nn.Sequential(Net(), torch.nn.ReLU())
        x = np.random.rand(1, 1, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, image_out=False)

    def test_unsupported_optimasation(self):
        class Reshape(Module):
            def forward(self, x):
                return x.reshape(4, 4, 16, 16)
        net = torch.nn.Sequential(GlobalAvgPool(), torch.nn.Linear(3, 4 * 16 * 16), Reshape(),
                                  torch.nn.Conv2d(4, 3, 3), torch.nn.ReLU())
        net.eval()
        x = np.random.rand(4, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x, missing_optimizations=True)

    def test_sqrt(self):
        class Sq(Module):
            def forward(self, x):
                return torch.sqrt(x)
        net = torch.nn.Sequential(Sq(), torch.nn.ReLU())
        x = np.random.rand(4, 3, 16, 16).astype(np.float32)
        convert_and_compare_output(net, x)



    # def test_inception_v3(self):
    #     net = models.Inception3(aux_logits=False)
    #     net.eval()
    #     x = np.random.rand(1, 3, 299, 299).astype(np.float32)
    #     convert_and_compare_output(net, x, image_out=False)



