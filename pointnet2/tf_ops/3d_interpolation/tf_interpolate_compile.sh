#/bin/bash
# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so \
    -shared -fPIC -I /home/wenchang/miniconda3/envs/cgn/lib/python3.7/site-packages/tensorflow/include \
    -I /home/wenchang/miniconda3/envs/cgn/pkgs/cuda-toolkit/include \
    -lcudart -L /home/wenchang/miniconda3/envs/cgn/pkgs/cuda-toolkit/lib64 \
    -I$TF_INC/external/nsync/public -l:libtensorflow_framework.so.2 -L$TF_LIB \
    -O2 -D_GLIBCXX_USE_CXX11_ABI=0