#include "DetDataset.h"
#include"utils/tinyxml.h"
#include "utils/readfile.h"

std::vector<BBox> loadXML(std::string xml_path)
{
	std::vector<BBox> objects;
	// XML文档
	TiXmlDocument doc;
	// 加载XML文档
	if (!doc.LoadFile(xml_path.c_str()))
	{
		std::cerr << doc.ErrorDesc() << std::endl;
		return objects;
	}

	// 定义根节点变量并赋值为文档的第一个根节点
	TiXmlElement *root = doc.FirstChildElement();
	// 如果没有找到根节点,说明是空XML文档或者非XML文档
	if (root == NULL)
	{
		std::cerr << "Failed to load file: No root element." << std::endl;
		// 清理内存
		doc.Clear();
		return objects;
	}

	// 遍历子节点
	for (TiXmlElement *elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
	{
		// 获取元素名
		std::string elemName = elem->Value();
		std::string name = "";
		// 获取元素属性值
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
	// 清理内存
	doc.Clear();
	return objects;
}

//遍历该目录下的.xml文件，并且找到对应的
void load_det_data_from_folder(std::string folder, std::string image_type,
	std::vector<std::string> &list_images, std::vector<std::string> &list_labels)
{
	for_each_file(folder,
		// filter函数，lambda表达式
		[&](const char*path, const char* name) {
		auto full_path = std::string(path).append({ file_sepator() }).append(name);
		std::string lower_name = tolower1(name);

		//判断是否为jpeg文件
		if (end_with(lower_name, ".xml")) {
			list_labels.push_back(full_path);
			std::string image_path = replace_all_distinct(full_path, ".xml", image_type);
			image_path = replace_all_distinct(image_path, "\\labels\\", "\\images\\");
			list_images.push_back(image_path);
		}
		//因为文件已经已经在lambda表达式中处理了，
		//不需要for_each_file返回文件列表，所以这里返回false
		return false;
	}
		, true//递归子目录
		);
}