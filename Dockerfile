FROM tensorflow/tensorflow:1.15.2-py3
RUN pip install torch torchvision
RUN pip install pytest
RUN pip install onnx
RUN mkdir /code
WORKDIR /code
