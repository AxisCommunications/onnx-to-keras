# onnx2keras

[![Travis CI][travis-image]][travis-url]


[travis-image]: https://travis-ci.com/AxisCommunications/onnx2keras?branch=master
[travis-url]: https://travis-ci.com/AxisCommunications/onnx2keras

This is a tool for converting onnx models as exported by for example pytorch into tensorflow keras 
models. It focuses on inference performance and what we call high-level-compatibility rather than 
completeness. That is, it will not be able to convert every onnx model, but the models it can convert 
it will convert in a good way. With high-level-compatibility we mean that the converted models produced
are constructed using the high-level keras API and should be similar to how the model would have 
been implemented in keras if implemented by hand in keras.

Installation
------------
```
    pip install onnx2keras
```

Usage
-----
```
    onnx2keras <infile.onnx> [<outfile.h5>]
```


