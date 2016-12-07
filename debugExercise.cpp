#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <vector>
#include <map>

#define POSITIVE_LABEL (0)
#define NEGATIVE_LABEL (1)

using namespace std;
using namespace cv;

typedef std::pair<int, cv::Mat> DatabaseElement;
typedef std::vector<DatabaseElement> DatabaseType;

// naming convention: [pos|neg]_[#].jpg
bool loadImages(const std::string& path, DatabaseType& outDatabase)
{
	cv::Mat posImage, negImage;
	int counter = 1;

	std::cout << "- Loading from: " << path << std::endl;

	do
	{
		posImage = cv::imread(path + "pos_" + std::to_string(counter) + ".jpg");
		negImage = cv::imread(path + "neg_" + std::to_string(counter) + ".jpg");

		if (!posImage.empty())
		{
			outDatabase.push_back(std::make_pair(POSITIVE_LABEL, posImage));
		}

		if (!negImage.empty())
		{
			outDatabase.push_back(std::make_pair(NEGATIVE_LABEL, negImage));
		}

		counter++;
	} while (!(posImage.empty() && negImage.empty()));

	std::cout << "- Number of images loaded: " << outDatabase.size() << std::endl;

	return !outDatabase.empty();
}

bool testSVM(const DatabaseType& trainingDb)
{
	CV_Assert(!trainingDb.empty());
	//CV_Assert(!vocabulary.empty());

	initModule_nonfree();

	Ptr<cv::FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	
	BOWImgDescriptorExtractor bowide(extractor, matcher);

	Mat vocabulary;
	FileStorage fs_vocabulary("vocabulary_surf.xml", FileStorage::READ);
	fs_vocabulary["vocabulary"] >> vocabulary;
	//fs_vocabulary.release();
	bowide.setVocabulary(vocabulary);

	CvSVM svm;
	svm.load("SVM_SURF_BOW.xml");

	int counter=0;

	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(it.second, keypoints);

		if (keypoints.empty()) continue;

		// Responses to the vocabulary
		cv::Mat imgDescriptor;
		bowide.compute(it.second, keypoints, imgDescriptor);

		if (imgDescriptor.empty()) continue;

		//cv::Mat results;
		float res = svm.predict(imgDescriptor);

		if (res != it.first){
			counter++;
			cout << "a  sample missclassified" << endl;
		}

	}

	cout << "误分类的数目：" << to_string(counter) << endl;
	float rate = (counter+0.0) / trainingDb.size();
	cout << "误分类的比率：" << rate << endl;

	return true;
}

int main()
{
	std::cout << "1. Loading images" << std::endl;
	const std::string trainingPath("D:/Qt Project/image/train image/train/New/");

	DatabaseType trainingDb;

	if (!loadImages(trainingPath, trainingDb))
	{
		return -1;
	}

	if (!testSVM(trainingDb))
	{
		return -1;
	}

	std::cout << std::endl;

	return 1;
}