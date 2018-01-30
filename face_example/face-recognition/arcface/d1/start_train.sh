#!/bin/bash
# Usage:
# ./start_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3
mkdir -p face_snapshot
../../../build/tools/caffe train -solver face_solver.prototxt  -gpu 0 2>&1 | tee result/addsoftmax_train_20180129_1.log
