
all:
	cython ../common/nms.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o ./lib/nms.so ../common/nms.c
	rm -rf ../common/nms.c