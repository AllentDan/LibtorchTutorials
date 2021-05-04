#include "DetDataset.h"
#include"utils/tinyxml.h"
#include "utils/readfile.h"

std::vector<BBox> loadXML(std::string xml_path)
{
	std::vector<BBox> objects;
	// XML�ĵ�
	TiXmlDocument doc;
	// ����XML�ĵ�
	if (!doc.LoadFile(xml_path.c_str()))
	{
		std::cerr << doc.ErrorDesc() << std::endl;
		return objects;
	}

	// ������ڵ��������ֵΪ�ĵ��ĵ�һ�����ڵ�
	TiXmlElement *root = doc.FirstChildElement();
	// ���û���ҵ����ڵ�,˵���ǿ�XML�ĵ����߷�XML�ĵ�
	if (root == NULL)
	{
		std::cerr << "Failed to load file: No root element." << std::endl;
		// �����ڴ�
		doc.Clear();
		return objects;
	}

	// �����ӽڵ�
	for (TiXmlElement *elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
	{
		// ��ȡԪ����
		std::string elemName = elem->Value();
		std::string name = "";
		// ��ȡԪ������ֵ
		if (strcmp(elemName.data(), "object") == 0)
		{
			for (TiXmlNode *object = elem->FirstChildElement(); object != NULL; object = object->NextSiblingElement())
			{
				if (strcmp(object->Value(), "name") == 0)
				{
					name = object->FirstChild()->Value();
				}

				if (strcmp(object->Value(), "bndbox") == 0)
				{
					BBox obj;
					TiXmlElement * xmin = object->FirstChildElement("xmin");
					TiXmlElement * ymin = object->FirstChildElement("ymin");
					TiXmlElement * xmax = object->FirstChildElement("xmax");
					TiXmlElement * ymax = object->FirstChildElement("ymax");

					obj.xmin = atoi(std::string(xmin->FirstChild()->Value()).c_str());
					obj.xmax = atoi(std::string(xmax->FirstChild()->Value()).c_str());
					obj.ymin = atoi(std::string(ymin->FirstChild()->Value()).c_str());
					obj.ymax = atoi(std::string(ymax->FirstChild()->Value()).c_str());
					obj.name = name;
					objects.push_back(obj);
				}

				//cout << bndbox->Value() << endl;
			}
		}
	}
	//std::cout << xml_path << std::endl;
	// �����ڴ�
	doc.Clear();
	return objects;
}

//������Ŀ¼�µ�.xml�ļ��������ҵ���Ӧ��
void load_det_data_from_folder(std::string folder, std::string image_type,
	std::vector<std::string> &list_images, std::vector<std::string> &list_labels)
{
	for_each_file(folder,
		// filter������lambda���ʽ
		[&](const char*path, const char* name) {
		auto full_path = std::string(path).append({ file_sepator() }).append(name);
		std::string lower_name = tolower1(name);

		//�ж��Ƿ�Ϊjpeg�ļ�
		if (end_with(lower_name, ".xml")) {
			list_labels.push_back(full_path);
			std::string image_path = replace_all_distinct(full_path, ".xml", image_type);
			image_path = replace_all_distinct(image_path, "\\labels\\", "\\images\\");
			list_images.push_back(image_path);
		}
		//��Ϊ�ļ��Ѿ��Ѿ���lambda���ʽ�д����ˣ�
		//����Ҫfor_each_file�����ļ��б��������ﷵ��false
		return false;
	}
		, true//�ݹ���Ŀ¼
		);
}