FROM tensorflow/tensorflow:1.14.0-py3
RUN pip install torch==1.2.0 torchvision==0.4.0
RUN pip install pytest
RUN pip install onnx
RUN mkdir /code
WORKDIR /code
