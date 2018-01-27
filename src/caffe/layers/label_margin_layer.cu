#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_margin_layer.hpp"

namespace caffe {


	template <typename Dtype>
	__global__ void LabelMarginForward(const int n, const int dim, const Dtype* label,
		Dtype* top_data, const Dtype* bottom_data,Dtype cos_m,const Dtype* sqrt_sin_data,Dtype sin_m) {
		CUDA_KERNEL_LOOP(index, n) {
			int gt = static_cast<int>(label[index]);
			if (bottom_data[index * dim + gt] > sin_m) 
				top_data[index * dim + gt] = bottom_data[index* dim+ gt]*cos_m - sqrt(sqrt_sin_data[index*dim+gt])*sin_m;
		}
	}

	template <typename Dtype>
	void LabelMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype cos_m = cos(bias_);
		Dtype sin_m = cos(bias_);

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		int num = bottom[0]->num();
		int count = bottom[0]->count();
		int dim = count / num; // 通常为1

		caffe_copy(count, bottom_data, top_data);
		caffe_sqr(count, bottom_data, squar_sin_data.mutable_gpu_data());
		caffe_set(count,Dtype(1), one_data.mutable_gpu_data());
		caffe_cpu_axpby(count, Dtype(1), one_data.gpu_data(), Dtype(-1), squar_sin_data.mutable_gpu_data());

		if (!transform_test_ && this->phase_ == TEST) return;

		// NOLINT_NEXT_LINE(whitespace/operators)
		LabelMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
			num, dim, label_data, top_data, bottom_data,cos_m,squar_sin_data.gpu_data(),sin_m);
		CUDA_POST_KERNEL_CHECK;
	}


	template <typename Dtype>
	__global__ void LabelMarginBackward(const int n, const int dim, const Dtype* label,
		Dtype* bottom_diff,  const Dtype* sqrt_sin_data, Dtype cos_m, const Dtype* bottom_data, Dtype sin_m) {
		CUDA_KERNEL_LOOP(index, n) {
			int gt = static_cast<int>(label[index]);
			if (bottom_data[index * dim + gt] > sin_m)
				bottom_diff[index * dim + gt] *= (-sqrt(sqrt_sin_data[index*dim+gt])*cos_m-bottom_data[index*dim+gt]*sin_m);
		}
	}


	template <typename Dtype>
	void LabelMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {

			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* bottom_data = bottom[0]->gpu_data();
            const Dtype* label_data = bottom[1]->gpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();


			int count = top[0]->count();
			int num = bottom[0]->num();
			int dim = count / num; // 通常为1
			Dtype cos_m = cos(bias_);
			Dtype sin_m = cos(bias_);

			caffe_copy(count, top_diff, bottom_diff);

			if (!transform_test_ &&this->phase_ == TEST)
				return;

			LabelMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
				num, dim, label_data, bottom_diff, squar_sin_data.gpu_data(), cos_m, bottom_data, sin_m);
			CUDA_POST_KERNEL_CHECK;

		}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(LabelMarginLayer);
} // namespace caffe
