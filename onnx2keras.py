import warnings
from functools import partial

import onnx
from onnx import numpy_helper
import tensorflow as tf
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethodV1


class Operations:
    def make_op(self, op_type, inputs, attrs):
        # print(op_type)
        # print([i.shape for i in inputs])
        # print(attrs)
        # print()
        return getattr(self, 'op_' + op_type.lower())(*inputs, **attrs)

class DataFormat: pass
class OnnxTensor(DataFormat): pass
class OnnxConstant(OnnxTensor): pass
class InterleavedImageBatch(DataFormat): pass

class OptimizationMissingWarning(Warning): pass

def ensure_data_format(tensor, format):
    if issubclass(tensor.data_format, format):
        return tensor
    elif tensor.data_format is OnnxConstant and format is InterleavedImageBatch:
        assert len(tensor.shape) == 4
        out = tensor.transpose([0, 2, 3, 1])
        out.data_format = InterleavedImageBatch
        return out
    elif tensor.data_format is OnnxTensor and format is InterleavedImageBatch:
        assert len(tensor.shape) == 4
        n, c, h, w = tensor.shape
        if h == w == 1 or c == 1:
            out = tf.reshape(tensor, [n, h, w, c])
        else:
            out = tf.transpose(tensor, [0, 2, 3, 1])
            warnings.warn("Transpose inserted. Please report at https://github.com/AxisCommunications/onnx-to-keras/issues", OptimizationMissingWarning)
        out.data_format = InterleavedImageBatch
        return out
    elif tensor.data_format is InterleavedImageBatch and format is OnnxTensor:
        assert len(tensor.shape) == 4
        n, h, w, c = tensor.shape
        if h == w == 1 or c == 1:
            out = tf.reshape(tensor, [n, c, h, w])
        else:
            out = tf.transpose(tensor, [0, 3, 1, 2])
            warnings.warn("Transpose inserted. Please report at https://github.com/AxisCommunications/onnx-to-keras/issues", OptimizationMissingWarning)
        out.data_format = OnnxTensor
        return out
    else:
        raise NotImplementedError

def compatible_data_format(format1, format2):
    return issubclass(format1, format2) or issubclass(format2, format1)

def ensure_compatible_data_format(a, b):
    if compatible_data_format(a.data_format, b.data_format):
        return a, b
    if b.data_format is OnnxConstant:
        return a, ensure_data_format(b, a.data_format)
    return ensure_data_format(a, b.data_format), b

class Constant(np.ndarray):
    data_format = OnnxConstant

class TfKerasOperations(Operations):
    keras = tf.keras

    def parse_attr(self, a):
        if a.type == onnx.AttributeProto.INT:
            return a.i
        elif a.type == onnx.AttributeProto.INTS:
            return tuple(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:
            return a.f
        elif a.type == onnx.AttributeProto.STRING:
            return a.s
        elif a.type == onnx.AttributeProto.TENSOR:
            return self.make_constant(numpy_helper.to_array(a.t))
        else:
            raise NotImplementedError

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
        weights = ensure_data_format(weights, OnnxConstant)  # XXX Assumes no ops on weights
        if len(kernel_shape) == 2:
            x = ensure_data_format(x, InterleavedImageBatch)
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
                bias = ensure_data_format(bias, OnnxConstant)  # XXX Assumes no ops on weights
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

    def op_leakyrelu(self, x, alpha):
        out = self.keras.layers.LeakyReLU(alpha=alpha)(x)
        out.data_format = x.data_format
        return [out]

    def op_sigmoid(self, x):
        out = self.keras.activations.sigmoid(x)
        out.data_format = x.data_format
        return [out]

    def op_softmax(self, x, axis):
        out = self.keras.activations.softmax(x, axis=axis)
        out.data_format = x.data_format
        return [out]

    def op_prelu(self, x, alpha):
        alpha = ensure_data_format(alpha, OnnxConstant)  # XXX Assumes no ops on alpha
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

    def op_maxpool(self, x, kernel_shape, pads, strides, ceil_mode=0):
        assert ceil_mode == 0
        if len(kernel_shape) == 2:
            x = ensure_data_format(x, InterleavedImageBatch)
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
        if all(t.data_format is InterleavedImageBatch for t in tensors):
            axis = (0, 3, 1, 2)[axis]
            out = self.keras.layers.Concatenate(axis)(list(tensors))
            out.data_format = InterleavedImageBatch
        elif all(t.data_format is OnnxConstant for t in tensors):
            out = self.make_constant(np.concatenate(tensors, axis))
        else:
            raise NotImplementedError
        return [out]

    def op_convtranspose(self, x, weights, bias=None, kernel_shape=None, strides=None, pads=None, dilations=None,
                         group=None, output_padding=(0, 0)):
        assert kernel_shape is not None
        assert strides is not None
        assert pads is not None
        assert dilations is not None
        assert group is not None
        weights = ensure_data_format(weights, OnnxConstant)  # XXX Assumes no ops on weights
        if bias is None:
            use_bias = False
            bias_initializer = None
        else:
            bias = ensure_data_format(bias, OnnxConstant)  # XXX Assumes no ops on weights
            use_bias = True

        if len(kernel_shape) == 2:
            x = ensure_data_format(x,  InterleavedImageBatch)
            assert kernel_shape == weights.shape[2:4]
            _, h_in, w_in, _ = x.shape
            h_out = (h_in - 1) * strides[0] - 2 * pads[0] + dilations[0] * (kernel_shape[0] - 1) + 1 + output_padding[0]
            w_out=(w_in - 1) * strides[1] - 2 * pads[1] + dilations[1] * (kernel_shape[1] - 1) + 1 + output_padding[1]


            if pads == (0,0,0,0):
                padding = 'valid'
            elif h_out == strides[0] * h_in and w_out == strides[1] * w_in and output_padding==(0,0):
                padding = 'same'
                output_padding = None  # output_padding overrides the padding argument in keras
            else:
                raise NotImplementedError
            # Tf; filter_height, filter_width, out_channels, in_channels
            # Torch: (in_channels, out_channels, kH, kW)
            weights = weights.transpose(2, 3, 1, 0)
            if group == 1:
                weights_initializer = self.keras.initializers.Constant(weights.view(np.ndarray))
                if use_bias:
                    bias_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))
                conv = self.keras.layers.Conv2DTranspose(weights.shape[2], kernel_shape, strides,
                                                         dilation_rate=dilations, padding=padding,
                                                         kernel_initializer=weights_initializer,
                                                         use_bias=use_bias, bias_initializer=bias_initializer,
                                                         output_padding=output_padding)
                out = conv(x)
            else:
                splits = tf.split(x, group, axis=-1)
                convolved_splits = []
                n = weights.shape[3] // group
                assert group * n == weights.shape[3]
                for i, split in enumerate(splits):
                    weights_initializer = self.keras.initializers.Constant(weights[:, :, :, i*n:(i+1)*n].view(np.ndarray))
                    if use_bias:
                        bias_initializer = self.keras.initializers.Constant(bias[i*n:(i+1)*n].view(np.ndarray))
                    conv = self.keras.layers.Conv2DTranspose(weights.shape[2], kernel_shape, strides,
                                                             dilation_rate=dilations, padding=padding,
                                                             kernel_initializer=weights_initializer,
                                                             use_bias=use_bias, bias_initializer=bias_initializer,
                                                             output_padding=output_padding)
                    convolved_splits.append(conv(split))
                out = tf.concat(convolved_splits, -1)

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
        x = ensure_data_format(x, OnnxTensor)
        out = x
        if isinstance(x, Constant):
            for ax in sorted(axes):
                out = np.expand_dims(out, ax).view(Constant)
            out.data_format = x.data_format
        else:
            for ax in sorted(axes):
                out = self.keras.backend.expand_dims(out, ax)
            out.data_format = OnnxTensor
        return [out]

    def op_clip(self, x, min, max):
        if min == 0:
            out = self.keras.layers.ReLU(max)(x)
        else:
            out = self.keras.backend.clip(x, min, max)
        out.data_format = x.data_format
        return [out]

    def op_add(self, x1, x2):
        x1, x2 = ensure_compatible_data_format(x1, x2)
        out = self.keras.layers.Add()([x1, x2])
        out.data_format = x1.data_format
        return [out]

    def op_sub(self, x1, x2):
        x1, x2 = ensure_compatible_data_format(x1, x2)
        out = self.keras.layers.Subtract()([x1, x2])
        out.data_format = x1.data_format
        return [out]

    def op_reducemean(self, x, axes, keepdims):
        x = ensure_data_format(x, InterleavedImageBatch)
        if axes == (2, 3) and keepdims == 0:
            out = self.keras.layers.GlobalAveragePooling2D()(x)
            out.data_format = OnnxTensor
        else:
            raise NotImplementedError

        return [out]

    def op_gemm(self, x, weights, bias, beta, transB, alpha):
        x = ensure_data_format(x, OnnxTensor)
        if beta == 1.0 and transB == 1 and alpha == 1.0:
            weights_initializer = self.keras.initializers.Constant(weights.view(np.ndarray).T)
            bias_initializer = self.keras.initializers.Constant(bias.view(np.ndarray))
            out = self.keras.layers.Dense(weights.shape[0], kernel_initializer=weights_initializer,
                                          bias_initializer=bias_initializer)(x)
            out.data_format = OnnxTensor
        else:
            raise NotImplementedError
        return [out]

    def op_pad(self, x, pads, mode, value=0.0):
        x = ensure_data_format(x, InterleavedImageBatch)
        if mode == b'constant' and len(pads) == 8:
            assert len(x.shape) * 2 == len(pads)
            if pads[0] == pads[1] == pads[4] == pads[5] == 0:
                # ((top_pad, bottom_pad), (left_pad, right_pad))
                if value == 0.0:
                    paddings = ((pads[2], pads[6]), (pads[3], pads[7]))
                    out = self.keras.layers.ZeroPadding2D(paddings)(x)
                else:
                    paddings = ((0,0), (pads[2], pads[6]), (pads[3], pads[7]), (0,0))
                    out = tf.pad(x, paddings, constant_values=value)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_averagepool(self, x, kernel_shape, pads, strides):
        x = ensure_data_format(x, InterleavedImageBatch)
        if len(x.shape) == 4:
            if pads == (0,0,0,0):
                padding = 'valid'
            else:
                raise NotImplementedError
            out = self.keras.layers.AveragePooling2D(kernel_shape, strides, padding)(x)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_globalaveragepool(self, x):
        x = ensure_data_format(x, InterleavedImageBatch)
        if len(x.shape) == 4:
            out = self.keras.backend.mean(x, axis=[1, 2], keepdims=True)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_flatten(self, x, axis):
        if axis == 1 and len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            out = self.keras.layers.Flatten()(x)
        else:
            raise NotImplementedError
        out.data_format = OnnxTensor
        return [out]

    def op_slice(self, x, starts, ends, axes=None, steps=None):
        if axes is None:
            axes = range(len(starts))
        if steps is None:
            steps = [1] * len(starts)
        if x.data_format is OnnxConstant:
            if axes != (0,):
                raise NotImplementedError
            out = self.make_constant(x[starts[0]:ends[0]:steps[0]])
        else:
            x = ensure_data_format(x, InterleavedImageBatch)
            if len(x.shape) != 4:
                raise NotImplementedError
            if len(axes) == 1 and starts[0] != ends[0]:
                if axes[0] == 0:
                    out = x[starts[0]:ends[0]:steps[0],:,:,:]
                elif axes[0] == 1:
                    out = x[:,:,:,starts[0]:ends[0]:steps[0]]
                elif axes[0] == 2:
                    out = x[:,starts[0]:ends[0]:steps[0],:,:]
                elif axes[0] == 3:
                    out = x[:,:,starts[0]:ends[0]:steps[0],:]
                else:
                    raise NotImplementedError
            elif tuple(axes) == (2,3) and starts[0] != ends[0] and starts[1] != ends[1]:
                out = x[:,starts[0]:ends[0]:steps[0],starts[1]:ends[1]:steps[1],:]
            else:
                raise NotImplementedError
            out.data_format = InterleavedImageBatch
        return [out]

    def op_constant(self, value):
        out = value
        out.data_format = OnnxConstant
        return [out]

    def op_shape(self, x):
        shape = list(map(int, x.shape))
        if x.data_format is InterleavedImageBatch:
            n, h, w, f = shape
            shape = [n, f, h, w]
        return [self.make_constant(shape)]

    def op_gather(self, x, indices, axis=0):
        x = ensure_data_format(x, OnnxConstant)
        if axis == 0:
            return [self.make_constant(x[indices])]
        else:
            raise NotImplementedError

    def op_cast(self, x, to):
        dtype = {
            0: None, # UNDEFINED
            1: np.float,
            2: np.uint8,
            3: np.int8,
            4: np.uint16,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            8: str,
            9: np.bool,
            10: np.float16,
            11: np.double,
            12: np.uint32,
            13: np.uint64,
            14: np.complex64,
            15: np.complex128,
            # // Non-IEEE floating-point format based on IEEE754 single-precision
            # // floating-point number truncated to 16 bits.
            # // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
            #BFLOAT16 = 16;
        }[to]
        if x.data_format is OnnxConstant:
            return [self.make_constant(x.astype(dtype))]
        else:
            out = self.keras.backend.cast(x, dtype)
            out.data_format = x.data_format
            return [out]

    def op_mul(self, a, b):
        if b.shape == ():
            a, b = b, a
        if a.shape == ():
            out = a * b
            out.data_format = b.data_format
            return [out]
        a, b = ensure_compatible_data_format(a, b)
        if a.data_format is OnnxConstant:
            return [self.make_constant(a * b)]
        else:
            out = tf.keras.layers.Multiply()([a, b])
            out.data_format = a.data_format
            return [out]

    def op_floor(self, x):
        x = ensure_data_format(x, OnnxConstant)
        return [self.make_constant(np.floor(x))]

    def op_div(self, a, b):
        a = ensure_data_format(a, OnnxConstant)
        b = ensure_data_format(b, OnnxConstant)
        return [self.make_constant(a / b)]

    def op_upsample(self, x, scales, mode=b'nearest'):
        if mode == b'nearest':
            return self.op_resize(x, None, scales, coordinate_transformation_mode=b'asymmetric', nearest_mode=b'floor')
        raise NotImplementedError

    def op_resize(self, x, roi, scales, sizes=None, *,
                  coordinate_transformation_mode=b"half_pixel", cubic_coeff_a=-0.75, exclude_outside=0,
                  extrapolation_value=0.0, mode=b"nearest", nearest_mode=b"round_prefer_floor"):
        assert sizes is None
        assert scales[0] == scales[1] == 1
        assert len(scales) == 4
        assert cubic_coeff_a == -0.75
        assert exclude_outside == 0
        assert extrapolation_value == 0.0

        x = ensure_data_format(x, InterleavedImageBatch)
        size = [x.shape[1] * int(scales[2]), x.shape[2] * int(scales[3])]
        if mode == b'nearest' and coordinate_transformation_mode == b'asymmetric' and nearest_mode==b'floor':
            out = tf.compat.v1.image.resize(x, size, ResizeMethodV1.NEAREST_NEIGHBOR)
        elif mode == b'linear' and coordinate_transformation_mode == b'align_corners':
            out = tf.compat.v1.image.resize(x, size, ResizeMethodV1.BILINEAR, align_corners=True)
        else:
            raise NotImplementedError
        out.data_format = InterleavedImageBatch
        return [out]

    def op_equal(self, x, y):
        x, y = ensure_compatible_data_format(x, y)
        out = self.keras.backend.equal(x, y)
        out.data_format = x.data_format
        return [out]

    def op_reshape(self, x, shape):
        x = ensure_data_format(x, OnnxTensor)
        assert x.shape[0] == shape[0]
        out = self.keras.layers.Reshape(shape[1:])(x)
        out.data_format = OnnxTensor
        return [out]

    def op_transpose(self, x, perm):
        x = ensure_data_format(x, OnnxConstant)
        x = x.transpose(perm)
        x.data_format = OnnxConstant
        return [x]

    def op_matmul(self, x1, x2):
        x1 = ensure_data_format(x1, OnnxTensor)
        x2 = ensure_data_format(x2, OnnxTensor)
        if x1.data_format is OnnxConstant:
            x1 = tf.convert_to_tensor(x1)
        if x2.data_format is OnnxConstant:
            x2 = tf.convert_to_tensor(x2)
        if len(x1.shape) == 2:
            assert len(x2.shape) == 2
            out = self.keras.backend.dot(x1, x2)
        elif len(x1.shape) == 3:
            assert len(x2.shape) == 3
            assert x1.shape[0] == x2.shape[0] == 1
            out = self.keras.backend.dot(x1, x2)
            out = tf.reshape(out, (1, out.shape[1], out.shape[3]))
        elif len(x1.shape) == 4:
            assert len(x2.shape) == 4
            assert x1.shape[0] == x2.shape[0] == 1
            assert x1.shape[1] == x2.shape[1] == 1
            out = self.keras.backend.dot(x1, x2)
            out = tf.reshape(out, (1, 1, out.shape[2], out.shape[5]))
        else:
            raise NotImplementedError
        out.data_format = OnnxTensor
        return [out]

    def op_sqrt(self, x):
        out = self.keras.backend.sqrt(x)
        out.data_format = x.data_format
        return [out]

    def op_abs(self, x):
        out = self.keras.backend.abs(x)
        out.data_format = x.data_format
        return [out]

    def op_neg(self, x):
        out = -x
        out.data_format = x.data_format
        return [out]



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
        attrs = {a.name: ops.parse_attr(a) for a in node.attribute}
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
        tf.keras.experimental.export_saved_model(model, export_saved_model)
    else:
        model.save(outfile)

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
