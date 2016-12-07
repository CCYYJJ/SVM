#include <map>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>

#define POSITIVE_LABEL (0)
#define NEGATIVE_LABEL (1)

using namespace cv;
using namespace std;

typedef std::pair<int, cv::Mat> DatabaseElement;
typedef std::vector<DatabaseElement> DatabaseType;
typedef vector<Mat> DataTest;

//const std::string image_folder_path("D:\Qt Project\image\train image");
//����ͼ��ŵ�outDatabase���������
bool loadTrainImages(const std::string& path, DatabaseType& outDatabase)
{
	cv::Mat posImage, negImage;
	int counter = 1;

	std::cout << "- Loading from: " << path << std::endl;

	do
	{
		posImage = cv::imread(path + "pos_" + std::to_string(counter) + ".jpg", 0);
		negImage = cv::imread(path + "neg_" + std::to_string(counter) + ".jpg", 0);

		if (!posImage.empty())
		{
			//����֮����Լ�һЩͼ�������
			//Mat Im = Mat::zeros(800, 900, CV_8UC3);
			//resize(posImage, posImage, Im.size());
			//medianBlur(posImage, posImage, 5);
			normalize(posImage, posImage, 0, 255, cv::NORM_MINMAX);
			//equalizeHist(posImage, posImage);
			//threshold(posImage, posImage,1 ,255,THRESH_OTSU);
			//Scharr(posImage, posImage, CV_32F, 1, 0);
			//adaptiveThreshold(posImage, posImage, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
			//medianBlur(posImage, posImage, 9);
			outDatabase.push_back(std::make_pair(POSITIVE_LABEL, posImage));
		}

		if (!negImage.empty())
		{
			//Mat Im = Mat::zeros(800, 900, CV_8UC3);
			//resize(negImage, negImage, Im.size());
			//medianBlur(negImage, negImage, 5);
			normalize(negImage, negImage, 0, 255, cv::NORM_MINMAX);
			//equalizeHist(posImage, posImage);
			//threshold(posImage, posImage, 1, 255, THRESH_OTSU);
			//adaptiveThreshold(negImage, negImage, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
			//medianBlur(negImage, negImage, 9);
			outDatabase.push_back(std::make_pair(NEGATIVE_LABEL, negImage));
		}

		counter++;
	} while (!(posImage.empty() && negImage.empty()));

	std::cout << "- Number of images loaded: " << outDatabase.size() << std::endl;//������ݴ�С

	return !outDatabase.empty();
}

bool loadTestImages(const std::string& path, DataTest& outDatabase){
	Mat test;
	int count = 1;
	std::cout << "- Loading from: " << path << std::endl;

	do
	{
		test = imread(path + std::to_string(count) + ".jpg", 0);
		if (!test.empty())
		{
			Mat Im = Mat::zeros(800, 900, CV_8UC3);
			resize(test, test, Im.size());
			//medianBlur(test, test, 5);
			normalize(test, test, 0, 255, cv::NORM_MINMAX);
			//adaptiveThreshold(test, test, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
			//equalizeHist(test, test);
			//threshold(test, test, 1, 255, THRESH_OTSU);
			//medianBlur(test, test, 9);
			outDatabase.push_back(test);
		}
		count++;
	} while (!test.empty());

	std::cout << "- Number of image test: " << outDatabase.size() << endl;
	return !outDatabase.empty();
}

//�����ʻ����������Ϊѵ�����ݺʹʻ��
bool createVocabulary(const DatabaseType& trainingDb, cv::Mat& outVocabulary)
{
	CV_Assert(!trainingDb.empty());

	//����surf�������������ȡ��
	int minHessian = 400;//SURF�㷨�е�hessian��ֵ
	SurfFeatureDetector detector(minHessian);
	SurfDescriptorExtractor extractor;

	//ѵ����������
	cv::Mat trainingDescriptors(1, extractor.descriptorSize(), extractor.descriptorType());

	outVocabulary.create(0, 1, CV_32FC1);

	//��ȡ���������������������������ӵ�trainingDescriptors��ȥ
	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector.detect(it.second, keypoints);

		if (!keypoints.empty())
		{
			cv::Mat descriptors;
			extractor.compute(it.second, keypoints, descriptors);

			if (!descriptors.empty())
			{
				std::cout << "- Adding " << descriptors.rows << " training descriptors." << std::endl;
				trainingDescriptors.push_back(descriptors);
			}
			else
			{
				std::cout << "- No descriptors found." << std::endl;
			}
		}
		else
		{
			std::cout << "- No keypoints found." << std::endl;
		}
	}

	if (trainingDescriptors.empty())
	{
		std::cout << "- Training descriptors are empty." << std::endl;
		return false;
	}
	//BOW�е�KMean���࣬ͨ������������Ĵʻ�
	BOWKMeansTrainer bowtrainer(50);
	bowtrainer.add(trainingDescriptors);
	outVocabulary = bowtrainer.cluster();

	/*FileStorage fs_descriptors("training_descriptors.xml", FileStorage::WRITE);
	fs_descriptors << "training_descriptors" << trainingDescriptors;
	fs_descriptors.release();*/

	FileStorage fs_vocabulary("vocabulary_surf.xml", FileStorage::WRITE);
	fs_vocabulary << "vocabulary" << outVocabulary;
	fs_vocabulary.release();
	return true;

}

//ͨ��֮ǰ�Ĵʻ����BOW��������ȡ����ѵ������ת��Ϊ�����ͱ�ǩ
bool scourTrainingSet(const DatabaseType& trainingDb, const cv::Mat& vocabulary, cv::Mat& outSamples, cv::Mat& outLabels)
{
	//ѵ�����ʹʻ㶼Ҫ�ǿ�
	CV_Assert(!trainingDb.empty());
	CV_Assert(!vocabulary.empty());

	Ptr<cv::FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	cv::Mat samples;
	//��������ͱ�ǩ
	outSamples.create(0, 1, CV_32FC1);
	outLabels.create(0, 1, CV_32SC1);

	for (auto& it : trainingDb)
	{
		vector<cv::KeyPoint> keypoints;
		detector->detect(it.second, keypoints);

		if (!keypoints.empty())
		{
			cv::Mat descriptors;
			//BOW�����������
			bowide.compute(it.second, keypoints, descriptors);

			if (!descriptors.empty())
			{
				if (samples.empty())
				{
					samples.create(0, descriptors.cols, descriptors.type());
				}

				// Copy class samples and labels
				std::cout << "- Adding " << descriptors.rows << " positive sample." << std::endl;
				samples.push_back(descriptors);//����������ӵ�samples��

				cv::Mat classLabels;

				//��ͼ��ı�ǩ�ŵ������ǩ��
				if (it.first == POSITIVE_LABEL)
				{
					classLabels = cv::Mat::zeros(descriptors.rows, 1, CV_32SC1);
				}
				else
				{
					classLabels = cv::Mat::ones(descriptors.rows, 1, CV_32SC1);
				}

				outLabels.push_back(classLabels);
			}
			else
			{
				std::cout << "- No descriptors found." << std::endl;
			}
		}
		else
		{
			std::cout << "- No keypoints found." << std::endl;
		}
	}

	if (samples.empty() || outLabels.empty())
	{
		std::cout << "- Samples are empty." << std::endl;
		return false;
	}

	//��sampleת��ΪoutSamples
	samples.convertTo(outSamples, CV_32FC1);

	return true;
}

//ѵ��֧��������
bool trainSVM(const cv::Mat& samples, const cv::Mat& labels)
{
	CV_Assert(!samples.empty() && samples.type() == CV_32FC1);
	CV_Assert(!labels.empty() && labels.type() == CV_32SC1);

	CvSVMParams svm_param;
	svm_param.svm_type = CvSVM::C_SVC;
	svm_param.kernel_type = CvSVM::RBF;
	//svm_param.nu = 0.5; // in the range 0..1, the larger the value, the smoother the decision boundary
	svm_param.C = 5;
	svm_param.gamma = 0.1;
	//svm_param.degree = 3;
	svm_param.term_crit.epsilon = 1e-8;
	svm_param.term_crit.max_iter = 1e9;
	svm_param.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	CvSVM outSVM;
	outSVM.train_auto(samples, labels, Mat(), Mat(), svm_param,
		15,outSVM.get_default_grid(CvSVM::C),outSVM.get_default_grid(CvSVM::GAMMA));
	outSVM.save("SVM_SURF_BOW.xml");

	return true;
}

//����֧��������
bool testSVM(const DataTest& trainingDb, const cv::Mat& vocabulary)
{
	CV_Assert(!trainingDb.empty());
	CV_Assert(!vocabulary.empty());

	Ptr<cv::FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	
	//FileStorage fs_vocabulary("vocabulary_surf.xml", FileStorage::READ);
	//fs_vocabulary << "vocabulary" << vocabulary;
	//fs_vocabulary.release();
	bowide.setVocabulary(vocabulary);

	CvSVM svm;
	svm.load("SVM_SURF_BOW.xml");
	int i = 1;

	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(it, keypoints);

		if (keypoints.empty()) continue;

		// Responses to the vocabulary
		cv::Mat imgDescriptor;
		bowide.compute(it, keypoints, imgDescriptor);

		if (imgDescriptor.empty()) continue;

		//cv::Mat results;
		float res = svm.predict(imgDescriptor);

		std::string predicted_label;
		if (res == POSITIVE_LABEL){
			predicted_label = "Have Crack";
		}
		else{
			predicted_label = "no Crack";
		}

		std::cout << i << "- Result of prediction: (" << predicted_label << "): " << res << std::endl;

		//cv::imshow(predicted_label, it);
		//cv::waitKey(-1);

		//cv::destroyWindow(predicted_label);
		i++;
	}

	return true;
}

int main(int argc, char *argv[])
{

	std::cout << "1. Loading images" << std::endl;
	const std::string trainingPath("D:/Qt Project/image/train image/train/New/");
	const std::string testingPath("D:/Qt Project/image/test image/");

	DatabaseType trainingDb; 
	DataTest testingDb;

	if (!loadTrainImages(trainingPath, trainingDb))
	{
		return -1;
	}

	std::cout << std::endl;

	// -------------------------------------------

	std::cout << "2. Creating vocabulary for BOW" << std::endl;

	cv::Mat vocabulary;
	if (!createVocabulary(trainingDb, vocabulary)){
		return -1;
	}

	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "3. Scour the training set for our histograms" << std::endl;

	cv::Mat samples_32f;
	cv::Mat labels;
	if (!scourTrainingSet(trainingDb, vocabulary, samples_32f, labels))
	{
		return -1;
	}

	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "4. Training SVM" << std::endl;

	if (!trainSVM(samples_32f, labels)){
		return -1;
	}

	std::cout << "Finish Training"<<std::endl;
	// -------------------------------------------

	/*std::cout << "5. Testing SVM" << std::endl;

	std::cout << "�������ͼ��" << endl;
	if (!loadTestImages(testingPath, testingDb))
	{
		return -1;
	}

	std::cout << std::endl;


	if (!testSVM(testingDb, vocabulary))
	{
		return -1;
	}

	std::cout << std::endl;*/
	// -------------------------------------------

	return 1;


}
