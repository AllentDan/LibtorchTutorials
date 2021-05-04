#include "util.h"
#include "readfile.h"

torch::Tensor DecodeBox(torch::Tensor input, torch::Tensor anchors, int num_classes, int img_size[])
{
	int num_anchors = anchors.sizes()[0];
	int bbox_attrs = 5 + num_classes;
	int batch_size = input.sizes()[0];
	int input_height = input.sizes()[2];
	int input_width = input.sizes()[3];
	//���㲽��
	//ÿһ���������Ӧԭ����ͼƬ�϶��ٸ����ص�
	//���������Ϊ13x13�Ļ���һ��������Ͷ�Ӧԭ����ͼƬ�ϵ�32�����ص�
	//416 / 13 = 32
	auto stride_h = img_size[1] / input_height;
	auto stride_w = img_size[0] / input_width;
	//�������ĳߴ�������������С����ʽ
	//�������������������϶�Ӧ�Ŀ��
	auto scaled_anchors = anchors.clone();
	scaled_anchors.select(1, 0) = scaled_anchors.select(1, 0) / stride_w;
	scaled_anchors.select(1, 1) = scaled_anchors.select(1, 1) / stride_h;

	//bs, 3 * (5 + num_classes), 13, 13->bs, 3, 13, 13, (5 + num_classes)
	//cout << "begin view"<<input.sizes()<<endl;
	auto prediction = input.view({ batch_size, num_anchors,bbox_attrs, input_height, input_width }).permute({ 0, 1, 3, 4, 2 }).contiguous();
	//cout << "end view" << endl;
	//����������λ�õĵ�������
	auto x = torch::sigmoid(prediction.select(-1, 0));
	auto y = torch::sigmoid(prediction.select(-1, 1));
	//�����Ŀ�ߵ�������
	auto w = prediction.select(-1, 2); // Width
	auto h = prediction.select(-1, 3); // Height

	//������Ŷȣ��Ƿ�������
	auto conf = torch::sigmoid(prediction.select(-1, 4));
	//�������Ŷ�
	auto pred_cls = torch::sigmoid(prediction.narrow(-1, 5, num_classes));// Cls pred.

	auto LongType = x.clone().to(torch::kLong).options();
	auto FloatType = x.options();

	//����������������ģ��������Ͻ� batch_size, 3, 13, 13
	auto grid_x = torch::linspace(0, input_width - 1, input_width).repeat({ input_height, 1 }).repeat(
		{ batch_size * num_anchors, 1, 1 }).view(x.sizes()).to(FloatType);
	auto grid_y = torch::linspace(0, input_height - 1, input_height).repeat({ input_width, 1 }).t().repeat(
		{ batch_size * num_anchors, 1, 1 }).view(y.sizes()).to(FloatType);

	//���������Ŀ��
	auto anchor_w = scaled_anchors.to(FloatType).narrow(1, 0, 1);
	auto anchor_h = scaled_anchors.to(FloatType).narrow(1, 1, 1);
	anchor_w = anchor_w.repeat({ batch_size, 1 }).repeat({ 1, 1, input_height * input_width }).view(w.sizes());
	anchor_h = anchor_h.repeat({ batch_size, 1 }).repeat({ 1, 1, input_height * input_width }).view(h.sizes());

	//�����������������������
	auto pred_boxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
	pred_boxes.select(-1, 0) = x + grid_x;
	pred_boxes.select(-1, 1) = y + grid_y;
	pred_boxes.select(-1, 2) = w.exp() * anchor_w;
	pred_boxes.select(-1, 3) = h.exp() * anchor_h;

	//���ڽ��������Ϊ�����416x416�Ĵ�С
	std::vector<int> scales{ stride_w, stride_h, stride_w, stride_h };
	auto _scale = torch::tensor(scales).to(FloatType);
	//cout << pred_boxes << endl;
	//cout << conf << endl;
	//cout << pred_cls << endl;
	pred_boxes = pred_boxes.view({ batch_size, -1, 4 }) * _scale;
	conf = conf.view({ batch_size, -1, 1 });
	pred_cls = pred_cls.view({ batch_size, -1, num_classes });
	auto output = torch::cat({ pred_boxes, conf, pred_cls }, -1);
	return output;
}

std::string replace_all_distinct(std::string str, const std::string old_value, const std::string new_value)
{
	for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
	{
		if ((pos = str.find(old_value, pos)) != std::string::npos)
		{
			str.replace(pos, old_value.length(), new_value);
		}
		else { break; }
	}
	return   str;
}

//������Ŀ¼�µ�.xml�ļ��������ҵ���Ӧ��
void load_seg_data_from_folder(std::string folder, std::string image_type,
	std::vector<std::string> &list_images, std::vector<std::string> &list_labels)
{
	for_each_file(folder,
		// filter������lambda���ʽ
		[&](const char*path, const char* name) {
		auto full_path = std::string(path).append({ file_sepator() }).append(name);
		std::string lower_name = tolower1(name);

		//�ж��Ƿ�Ϊjpeg�ļ�
		if (end_with(lower_name, ".json")) {
			list_labels.push_back(full_path);
			std::string image_path = replace_all_distinct(full_path, ".json", image_type);
			list_images.push_back(image_path);
		}
		//��Ϊ�ļ��Ѿ��Ѿ���lambda���ʽ�д����ˣ�
		//����Ҫfor_each_file�����ļ��б��������ﷵ��false
		return false;
	}
		, true//�ݹ���Ŀ¼
		);
}

