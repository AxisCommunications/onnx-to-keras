test1:
	docker build --build-arg https_proxy --build-arg http_proxy -t onnx2keras .
	docker run -ti -v `pwd`:/code onnx2keras py.test -v
test2:
	docker build --build-arg https_proxy --build-arg http_proxy -t onnx2keras_tf2 -f Dockerfile.tf2 .
	docker run -ti -v `pwd`:/code onnx2keras_tf2 py.test -v

test: test1 test2
