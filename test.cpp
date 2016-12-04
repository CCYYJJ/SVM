#include "Hist.h"
#include "division.h"

int main()
{
	/*Mat posImage = imread("D:/Qt Project/image/train image/train/neg_3.jpg", 0);
	Mat Im = Mat::zeros(600, 900, CV_8UC3);
	resize(posImage, posImage, Im.size());
	medianBlur(posImage, posImage, 5);
	normalize(posImage, posImage, 0, 255, cv::NORM_MINMAX);
	equalizeHist(posImage, posImage);
	threshold(posImage, posImage, 1, 255, THRESH_OTSU);
	//adaptiveThreshold(posImage, posImage, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 3);
	//medianBlur(posImage, posImage, 9);

	imshow("output", posImage);
	waitKey(-1);
	return 1;*/
	Mat image = imread("D:/Qt Project/image/train image/train/neg_11.jpg", 0);
	//Mat Im = Mat::zeros(800, 900, CV_8UC3);
	//resize(image, image, Im.size());
	//medianBlur(image, image, 5);
	normalize(image, image, 0, 255, cv::NORM_MINMAX);
	//adaptiveThreshold(test, test, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
	//equalizeHist(image, image);
	/*Historgram1D h;
	Mat histo = h.getHistogram(image);
	for (int i = 0; i < 256;i++){
		cout << "value" << i << "=" << histo.at<float>(i) << endl;
	}
	imshow("Histogram", h.getHistogramImage(image));
	imshow("原图",image);
	Mat stretch = h.stretch(image, 0.01f);
	imshow("伸展图", stretch);
	imshow("伸展后直方图", h.getHistogramImage(stretch));*/
	vector<Mat> out;
	vector<Rect> rec;

	if (division(image,out,rec))
	{
		cout << "EORROR" << endl;
		return -1;
	}
	
	cout << "显示图片：" << endl;
	//int i = 0;
	vector<Rect>::iterator it1 = rec.begin();
	for (vector<Mat>::iterator itc = out.begin(); itc != out.end(); itc++){
			if (predictImage(*itc)){
				cout << "here have a crack" << endl;
				rectangle(image, *it1, Scalar(0, 255, 0));
			}
			it1++;
	}
	//int len = out.size();
	/*for (vector<Mat>::iterator itc = out.begin(); itc != out.end(); itc++){
		if (predictImage(*itc)){
			cout << "here have a crack" << endl;
			//imshow("out", *itc);
			//waitKey(-1);
			//destroyWindow("out");
		}
		//it1++;
	}*/
	
	imshow("out", image);
	waitKey(-1);

	return 1;
}