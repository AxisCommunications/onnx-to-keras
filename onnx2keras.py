from functools import partial

import onnx
from onnx import numpy_helper
import tensorflow as tf
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import numpy as np


class Operations:
    def make_op(self, op_type, inputs, attrs):
        # print(op_type)
        # print([i.shape for i in inputs])
        # print(attrs)
        # print()
        return getattr(self, 'op_' + op_type.lower())(*inputs, **attrs)

class OnnxConstant: pass
class InterleavedImageBatch: pass
class VectorBatch: pass

class Constant(np.ndarray):
    data_format = OnnxConstant

class TfKerasOperations(Operations):
    keras = tf.keras

    def make_constant(self, x):
        return np.asarray(x).view(Constant)

    def make_input(self, shape, dtype, name=None):
        dtype = tf.as_dtype(dtype)
        # XXX: Assumes all inputs are image batches that we want to transpose
        assert len(shape) == 4
        tensor = tf.keras.layers.Input((shape[2], shape[3], shape[1]), shape[0], name, dtype)
        tensor.data_format = InterleavedImageBatch
        return tensor

    def op_conv(self, x, weights, bias=None, kernel_shape=None, strides=None, pads=None, dilations=None, group=None):
        assert weights.data_format is OnnxConstant # XXX Assumes no ops on weights
        assert bias is None or bias.data_format is OnnxConstant # XXX Assumes no ops on weights
        if len(kernel_shape) == 2:
            assert x.data_format is InterleavedImageBatch
            assert kernel_shape == weights.shape[2:4]
            if group == 1:
                ConvClass = self.keras.layers.Conv2D
            elif group == x.shape[3]:
                def ConvClass(filters, kernel_size, strides, dilation_rate, padding,
                              kernel_initializer, use_bias=True, bias_initializer='zeros'):
                    return self.keras.layers.DepthwiseConv2D(kernel_size, strides, dilation_rate=dilation_rate,
                                                             padding=padding, use_bias=use_bias,
                                                             bias_initializer=bias_initializer,
                                                             depthwise_initializer=kernel_initializer)
            else:
                raise NotImplementedError
            if pads == (0,0,0,0):
                padding = 'valid'
            elif (kernel_shape[0] == kernel_shape[1] and pads[0] == pads[1] == pads[2] == pads[3] and
                  pads[0] * 2 + 1 == kernel_shape[0] and strides == (1, 1) and dilations == (1, 1)):
                padding = 'same'
            elif (kernel_shape == (3, 3) and pads == (1,1,1,1) and  strides == (2,2) and dilations == (1, 1) and
                  x.shape[1] % 2 == 1 and x.shape[2] % 2 == 1):
                padding = 'same'
            else:
                # ((top_pad, bottom_pad), (left_pad, right_pad))
                pad = self.keras.layers.ZeroPadding2D(((pads[0], pads[2]), (pads[1], pads[3])))
                x = pad(x)
                padding = 'valid'

            # Tf; filter_height, filter_width, in_channels, out_channels
            # Torch: (out_channels, in_channels, kH, kW)
            weights = weights.transpose(2, 3, 1, 0)
            weights_initializer = self.keras.initializers.Constant(weights.view(np.ndarray))
            if bias is None:
                conv = ConvClass(weights.shape[3], kernel_shape, strides,
                                 dilation_rate=dilations, padding=padding,
                                 kernel_initializer=weights_initializer, use_bias=False)
            else:
                bias_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))
                conv = ConvClass(weights.shape[3], kernel_shape, strides,
                                 dilation_rate=dilations, padding=padding,
                                 kernel_initializer=weights_initializer, bias_initializer=bias_initializer)
            out = conv(x)
            out.data_format = InterleavedImageBatch
            return [out]
        else:
            raise NotImplementedError

    def op_relu(self, x):
        out = self.keras.layers.ReLU()(x)
        out.data_format = x.data_format
        return [out]

    def op_prelu(self, x, alpha):
        assert alpha.data_format is OnnxConstant # XXX Assumes no ops on alpha
        if len(alpha) == 1:
            shared = list(range(1, len(x.shape)))
            alpha = alpha.reshape((1,) * (len(x.shape) - 1))
        elif len(alpha) == x.shape[-1]:
            shared = list(range(1, len(x.shape) - 1))
        else:
            raise NotImplementedError
        alpha_initializer = self.keras.initializers.Constant(alpha.view(np.ndarray))
        out = self.keras.layers.PReLU(shared_axes=shared, alpha_initializer=alpha_initializer)(x)
        out.data_format = x.data_format
        return [out]

    def op_maxpool(self, x, kernel_shape, pads, strides):
        if len(kernel_shape) == 2:
            assert x.data_format is InterleavedImageBatch
            if pads == (0, 0, 0, 0):
                padding = 'valid'
            else:
                # ((top_pad, bottom_pad), (left_pad, right_pad))
                pad = self.keras.layers.ZeroPadding2D(((pads[0], pads[2]), (pads[1], pads[3])))
                x = pad(x)
                padding = 'valid'
            out = self.keras.layers.MaxPool2D(kernel_shape, strides, padding)(x)
            out.data_format = InterleavedImageBatch
            return [out]
        else:
            raise NotImplementedError

    def op_concat(self, *tensors, axis):
        axis = (0, 3, 1, 2)[axis]
        out = self.keras.layers.Concatenate(axis)(list(tensors))
        for t in tensors:
            assert tensors[0].data_format is t.data_format
        out.data_format = tensors[0].data_format
        return [out]

    def op_convtranspose(self, x, weights, bias=None, kernel_shape=None, strides=None, pads=None, dilations=None, group=None):
        assert kernel_shape is not None
        assert strides is not None
        assert pads is not None
        assert dilations is not None
        assert group is not None
        assert weights.data_format is OnnxConstant # XXX Assumes no ops on weights
        if bias is None:
            use_bias = False
            bias_initializer = None
        else:
            assert bias.data_format is OnnxConstant # XXX Assumes no ops on weights
            use_bias = True
            bias_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))

        if len(kernel_shape) == 2:
            assert x.data_format is InterleavedImageBatch
            assert kernel_shape == weights.shape[2:4]
            if group != 1:
                raise NotImplementedError
            _, h_in, w_in, _ = x.shape
            h_out = (h_in - 1) * strides[0] - 2 * pads[0] + dilations[0] * (kernel_shape[0] - 1) + 1 # FIXME: + output_padding[0]
            w_out=(w_in - 1) * strides[1] - 2 * pads[1] + dilations[1] * (kernel_shape[1] - 1) + 1 # FIXME: + output_padding[1]

            if pads == (0,0,0,0):
                padding = 'valid'
            elif h_out == strides[0] * h_in and w_out == strides[1] * w_in:
                padding = 'same'
            else:
                raise NotImplementedError
            # Tf; filter_height, filter_width, out_channels, in_channels
            # Torch: (in_channels, out_channels, kH, kW)
            weights = weights.transpose(2, 3, 1, 0)
            weights_initializer = self.keras.initializers.Constant(weights.view(np.ndarray))
            conv = self.keras.layers.Conv2DTranspose(weights.shape[2], kernel_shape, strides,
                                                     dilation_rate=dilations, padding=padding,
                                                     kernel_initializer=weights_initializer,
                                                     use_bias=use_bias, bias_initializer=bias_initializer)
            out = conv(x)
            assert out.shape[1] == h_out
            assert out.shape[2] == w_out
            out.data_format = InterleavedImageBatch
            return [out]
        else:
            raise NotImplementedError

    def op_batchnormalization(self, x, weight, bias, running_mean, running_var, momentum, epsilon):
        if  len(x.shape) != 4:
            raise NotImplementedError
        moving_mean_initializer = self.keras.initializers.Constant(running_mean.view(np.ndarray))
        moving_variance_initializer = self.keras.initializers.Constant(running_var.view(np.ndarray))
        beta_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))
        gamma_initializer = self.keras.initializers.Constant(weight.view(np.ndarray))
        norm = self.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon,
                                                    moving_mean_initializer=moving_mean_initializer,
                                                    moving_variance_initializer=moving_variance_initializer,
                                                    beta_initializer=beta_initializer,
                                                    gamma_initializer=gamma_initializer)
        out = norm(x)
        out.data_format = x.data_format
        return [out]

    def op_unsqueeze(self, x, axes):
        if isinstance(x, Constant):
            for ax in axes:
                out = np.expand_dims(x, ax).view(Constant)
                out.data_format = x.data_format
        else:
            for ax in axes:
                out = self.keras.layers.Lambda(lambda x: self.keras.backend.expand_dims(x, ax))(out)
            out.data_format = None
        return [out]

    def op_clip(self, x, min, max):
        if min == 0:
            clip = self.keras.layers.ReLU(max)
        else:
            clip = self.keras.layers.Lambda(lambda x: self.keras.backend.clip(x, min, max))
        out = clip(x)
        out.data_format = x.data_format
        return [out]

    def op_add(self, x1, x2):
        assert x1.data_format == x2.data_format
        out = self.keras.layers.Add()([x1, x2])
        out.data_format = x1.data_format
        return [out]

    def op_reducemean(self, x, axes, keepdims):
        assert x.data_format is InterleavedImageBatch
        if axes == (2, 3) and keepdims == 0:
            out = self.keras.layers.GlobalAveragePooling2D()(x)
            out.data_format = VectorBatch
        else:
            raise NotImplementedError

        return [out]

    def op_gemm(self, x, weights, bias, beta, transB, alpha):
        assert x.data_format is VectorBatch
        if beta == 1.0 and transB == 1 and alpha == 1.0:
            weights_initializer = self.keras.initializers.Constant(weights.view(np.ndarray).T)
            bias_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))
            out = self.keras.layers.Dense(weights.shape[0], kernel_initializer=weights_initializer,
                                          bias_initializer=bias_initializer)(x)
            out.data_format = VectorBatch
        else:
            raise NotImplementedError
        return [out]

def parse_attr(a):
    if a.type == onnx.AttributeProto.INT:
        return a.i
    elif a.type == onnx.AttributeProto.INTS:
        return tuple(a.ints)
    elif a.type == onnx.AttributeProto.FLOAT:
        return a.f
    elif a.type == onnx.AttributeProto.STRING:
        return a.s
    else:
        raise NotImplementedError


def onnx2keras(onnx_model):
    tensors = {}
    ops = TfKerasOperations()

    for init in onnx_model.graph.initializer:
        tensors[init.name] = ops.make_constant(numpy_helper.to_array(init))

    model_inputs = []
    for input in onnx_model.graph.input:
        if input.name in tensors:
            continue
        shape = [d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                 for d in input.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]
        tensors[input.name] = ops.make_input(shape, dtype, input.name)
        model_inputs.append(tensors[input.name])

    for node in onnx_model.graph.node:
        inputs = [tensors[i] for i in node.input]
        attrs = {a.name: parse_attr(a) for a in node.attribute}
        output_tensors = ops.make_op(node.op_type, inputs, attrs)
        assert len(output_tensors) == len(node.output)
        for n, t in zip(node.output, output_tensors):
            tensors[n] = t

    outputs = [tensors[o.name] for o in onnx_model.graph.output]
    return tf.keras.models.Model(model_inputs, outputs)

def main(infile, outfile=None, export_saved_model=False):
    if outfile is None:
        outfile = infile[:-5] if infile[-5:] == '.onnx' else infile
        outfile += '.h5'
    model = onnx2keras(onnx.load(infile))
    if export_saved_model:
        tf.keras.experimental.export_saved_model(model, "tst.tf")
    else:
        model.save(outfile)

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
