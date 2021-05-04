#include "Augmentations.h"

template<typename T>
T RandomNum(T _min, T _max)
{
	T temp;
	if (_min > _max)
	{
		temp = _max;
		_max = _min;
		_min = temp;
	}
	return rand() / (double)RAND_MAX *(_max - _min) + _min;
}

int BBox::GetH()
{
	return ymax - ymin;
}

int BBox::GetW()
{
	return xmax - xmin;
}

float BBox::CenterX()
{
	return (xmax + xmin) / 2.0;
}

float BBox::CenterY()
{
	return  (ymax + ymin) / 2.0;
}


Data Augmentations::Resize(Data mData, int width, int height, float probability) {
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) {

		float h_scale = height * 1.0 / mData.image.rows;
		float w_scale = width * 1.0 / mData.image.cols;
		for (int i = 0; i < mData.bboxes.size(); i++)
		{
			mData.bboxes[i].xmin = int(w_scale*mData.bboxes[i].xmin);
			mData.bboxes[i].xmax = int(w_scale*mData.bboxes[i].xmax);
			mData.bboxes[i].ymin = int(h_scale*mData.bboxes[i].ymin);
			mData.bboxes[i].ymax = int(h_scale*mData.bboxes[i].ymax);
		}

		cv::resize(mData.image, mData.image, cv::Size(width, height));

	}
	return mData;
}