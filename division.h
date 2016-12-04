#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <vector>

#define POSITIVE_LABEL (0)
#define NEGATIVE_LABEL (1)

using namespace std;
using namespace cv;

typedef vector<Mat> DataTest;

bool predictImage(Mat& src){
	CV_Assert(!src.empty());

	initModule_nonfree();
	
	Ptr<cv::FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(extractor, matcher);


	Mat vocabulary;
	FileStorage fs_vocabulary("vocabulary_surf.xml", FileStorage::READ);
	fs_vocabulary["vocabulary"]>> vocabulary;
	//fs_vocabulary.release();
	bowide.setVocabulary(vocabulary);

	CvSVM svm;
	svm.load("SVM_SURF_BOW.xml");

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(src, keypoints);

	/*if (keypoints.empty()){
		return false;
	}*/

	// Responses to the vocabulary
	cv::Mat imgDescriptor;
	bowide.compute(src, keypoints, imgDescriptor);

	/*if (imgDescriptor.empty()){
		return false;
	}*/

	//cv::Mat results;
	float res = svm.predict(imgDescriptor);

	if (res == POSITIVE_LABEL){
		cout << "have crack" << endl;
		return true;
	}
	else{
		cout << "no crack" << endl;
		return false;
	}
}

bool division(Mat& src,vector<Mat>& outMask,vector<Rect>& rec){
	const int nsize(200);
	int n1 = src.rows / nsize;
	int n2 = src.cols / nsize;
	Rect rect(0, 0, nsize, nsize);
	Mat patch;
	for (int j = 0; j < n1+1; j++){
		for (int i = 0; i < n2 + 1; i++){
			if (i == n2&&j==n1 ){
				rect.x = src.cols - nsize;
				rect.y = src.rows - nsize;
				Mat patch(src,rect);
				/*if (predictImage(patch)){
					rectangle(src, rect, Scalar(0, 255, 0), 1, 8, 0);
				}*/
				outMask.push_back(patch);
				rec.push_back(rect);
			}
			else if (i == n2){
				rect.x = src.cols - nsize;
				rect.y = j*nsize;
				Mat patch(src, rect);
				/*if (predictImage(patch)){
					rectangle(src, rect, Scalar(0, 255, 0), 1, 8, 0);
				}*/
				outMask.push_back(patch);
				rec.push_back(rect);
			}
			else if(j==n1){
				rect.x = i*nsize;
				rect.y = src.rows - nsize;
				Mat patch(src, rect);
				/*if (predictImage(patch)){
					rectangle(src, rect, Scalar(0, 255, 0), 1, 8, 0);
				}*/
				outMask.push_back(patch);
				rec.push_back(rect);
			}
			else{
				rect.x = i*nsize;
				rect.y = j*nsize;
				Mat patch(src, rect);
				/*if (predictImage(patch)){
					rectangle(src, rect, Scalar(0, 255, 0), 1, 8, 0);
				}*/
				outMask.push_back(patch);
				rec.push_back(rect);
			}
		}
	}
	cout << "输出图片数：" << outMask.size() << endl;
	cout << "输出矩形数：" << rec.size() << endl;
	//Rect rect(0, 0, nsize, nsize);


	return outMask.empty();
}
