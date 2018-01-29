#ifndef _CAFFE_LABEL_MARGIN_LAYER_
#define _CAFFE_LABEL_MARGIN_LAYER_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace caffe{
	template <typename Dtype>
	class LabelMarginLayer :public Layer < Dtype > {
	public:
		explicit LabelMarginLayer(const LayerParameter& param) :Layer<Dtype>(param){}

		virtual void LayerSetUp(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top);
		virtual void Reshape(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top);
		virtual inline const char* type() const { return "LabelMargin"; }
		virtual inline const int MinNumBottomBlobs(){ return 2; }

	private:
		virtual void Forward_cpu(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype> *>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *>& top, 
			const vector<bool>& propagate_down, const vector<Blob<Dtype> *>& bottom);

		Dtype bias_;
		bool transform_test_;
		//Blob<Dtype> squar_sin_data;
		//Blob<Dtype> one_data;
	};
}// namespace caffe

#endif
