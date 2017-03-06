#From inside tensorflow/tensorflow/iv_adas_pd_lib, run bazel build :ivadaspd
#libivadaspd.so will be created in tensorflow/bazel-bin/tensorflow/iv_adas_pd_lib

cc_library(
	name = "ivadaspd",
    srcs = ["iv_adas_pd.cc"],
    hdrs = ["iv_adas_pd.h"],
    deps = ["//tensorflow/core:tensorflow"],
)

#cc_binary(
#    name = "mnistpredict",
#    srcs = ["mnist.cc", "MNIST.h"],
#    deps = [
#        "//tensorflow/core:tensorflow",
#   ],
#)
