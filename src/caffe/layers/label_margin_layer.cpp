#include <algorithm>
#include <vector>
#include <caffe/layers/label_margin_layer.hpp>

namespace caffe{

	template <typename Dtype>
	void LabelMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *>& bottom,
		const vector<Blob<Dtype> *>& top){
		const LabelMarginParameter label_margin_param = this->layer_param_.label_margin_param();
		bias_ = label_margin_param.bias();
		transform_test_ = label_margin_param.transform_test() & (this->phase_ == TRAIN);
		
		CHECK_GT(bias_, 0);
		CHECK_LT(bias_, M_PI / 2);
	}

	template <typename Dtype>
	void LabelMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom,
		const vector<Blob<Dtype> *>& top){
		top[0]->ReshapeLike(*bottom[0]);
		squar_sin_data.ReshapeLike(*bottom[0]);
		one_data.ReshapeLike(*bottom[0]);
	}


	// 归一化后是cosθ 需要转化为cos(θ+m) = cosθcosm-sinθsinm  当然cos
	template <typename Dtype>
	void LabelMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom,
		const vector<Blob<Dtype> *>& top){
		Dtype cos_m = cos(bias_);
		Dtype sin_m = cos(bias_);

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int num = bottom[0]->num();
		int count = bottom[0]->count();
		int dim = count / num; // 通常为1

		caffe_copy(count, bottom_data, top_data);
		caffe_sqr(count, bottom_data, squar_sin_data.mutable_cpu_data());
		caffe_set(count, Dtype(1), one_data.mutable_cpu_data());
		caffe_cpu_axpby(count, Dtype(1), one_data.cpu_data(), Dtype(-1), squar_sin_data.mutable_cpu_data());


		if (!transform_test_ &&this->phase_ == TEST)
			return;
		for (int i = 0; i < num; i++){
			int gt = static_cast<int>(label_data[i]);
			if (bottom_data[i*dim+gt] > sin_m){
				top_data[i*dim + gt] = bottom_data[i*dim + gt] * cos_m - sqrt(squar_sin_data.cpu_data()[i*dim+gt])*sin_m;
			}
		}
	}

	template <typename Dtype>
	void LabelMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype> *>& bottom){

		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* label_data = bottom[1]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();


			int count = top[0]->count();
			int num = bottom[0]->num();
			int dim = count / num; // 通常为1
			Dtype cos_m = cos(bias_);
			Dtype sin_m = cos(bias_);

			caffe_copy(count, top_diff, bottom_diff);

			if (!transform_test_ &&this->phase_ == TEST)
				return;

			for (int i = 0; i < num; i++){
				int gt = static_cast<int>(label_data[i]);
				if (bottom_data[i*dim + gt] > sin_m){
					bottom_diff[i*dim + gt] *= (-sqrt(squar_sin_data.cpu_data()[i*dim + gt]) * cos_m - bottom_data[i*dim+gt]*sin_m);
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(LabelMarginLayer)
#endif

	INSTANTIATE_CLASS(LabelMarginLayer);
	REGISTER_LAYER_CLASS(LabelMargin);
} // namespace caffe
