#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat_<float> convolutionCorrelation(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel, bool convolution){
    int kernelRow_cnt = kernel.rows;
    int kernelCol_cnt = kernel.cols;
    int srcRow_cnt = src.rows;
    int srcCol_cnt = src.cols;
    int outPixRow = kernelRow_cnt/2;
    int outPixCol = kernelCol_cnt/2;

    if(convolution)
        cv::flip(kernel, kernel, 0);
    
    cv::Mat_<float> res(src.size(), CV_32FC1);
    for (int row_cnt = 0; row_cnt < srcRow_cnt; row_cnt++)
    {
        for (int col_cnt = 0; col_cnt < srcCol_cnt; col_cnt++)
        {
            float sum = 0;
            int rowTemp = 0, colTemp = 0;
            for (int kRow_cnt = 0; kRow_cnt < kernelRow_cnt; kRow_cnt++)
            {
                for (int kCol_cnt = 0; kCol_cnt < kernelCol_cnt; kCol_cnt++)
                {
                    float pixVal = 0;
                    colTemp = -outPixCol + kCol_cnt + col_cnt;
                    rowTemp = -outPixRow + kRow_cnt + row_cnt;
                    if (rowTemp < 0 || rowTemp > (srcRow_cnt-1) || colTemp < 0 || colTemp > (srcCol_cnt-1));
                    else
                    {
                        pixVal = src(row_cnt - outPixRow + kRow_cnt, col_cnt-outPixCol+kCol_cnt);
                    }
                    sum += kernel(kRow_cnt, kCol_cnt) * pixVal;
                }
            }
            res(row_cnt, col_cnt) = sum;
        }
    }
    return res.clone();
}

cv::Mat_<float> shrinkImage(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel, bool convolution){
    int kernelRow_cnt = kernel.rows;
    int kernelCol_cnt = kernel.cols;
    int srcRow_cnt = src.rows;
    int srcCol_cnt = src.cols;
    int outPixRow = kernelRow_cnt/2;
    int outPixCol = kernelCol_cnt/2;

    if(convolution)
        cv::flip(kernel, kernel, 0);
    
    int rowRes = 0, colRes = 0;

    cv::Mat_<float> res(src.size()/kernelRow_cnt, CV_32FC1);
    for (int row_cnt = 0; row_cnt < srcRow_cnt-kernelRow_cnt; row_cnt+=kernelRow_cnt)
    {
        for (int col_cnt = 0; col_cnt < srcCol_cnt-kernelRow_cnt; col_cnt+=kernelRow_cnt)
        {
            float sum = 0;
            for (int kRow_cnt = 0; kRow_cnt < kernelRow_cnt; kRow_cnt++)
            {
                for (int kCol_cnt = 0; kCol_cnt < kernelCol_cnt; kCol_cnt++)
                {
                    sum += kernel(kRow_cnt, kCol_cnt) * src(row_cnt + kRow_cnt, col_cnt + kCol_cnt);
                }
            }
            res(rowRes, colRes) = sum;
            colRes++;
        }
        colRes = 0;
        rowRes++;
    }
    return res.clone();
}

int main(int argc, char** argv ) {
    string fileName = "./Lenna.png";
    Mat_<float> smooth = Mat::ones(3, 3, CV_32FC1)/9.0f;
    Mat_<float> sharpen = Mat::zeros(3, 3, CV_32FC1);

    sharpen(0,1) = -1;
    sharpen(1,0) = -1;
    sharpen(1,1) = 5;
    sharpen(1,2) = -1;
    sharpen(2,1) = -1;

    Mat img = imread(fileName, IMREAD_GRAYSCALE);
    cv::imshow("source", img);
    Mat resConvolution = convolutionCorrelation(img, sharpen, true);
    resConvolution.convertTo(resConvolution, IMREAD_GRAYSCALE);
    cv::imshow("Convolution", resConvolution);
    Mat resCorrelation = convolutionCorrelation(img, sharpen, false);
    resCorrelation.convertTo(resCorrelation, IMREAD_GRAYSCALE);
    cv::imshow("Correlation", resCorrelation);
    
    Mat shrink = shrinkImage(img, smooth, false);
    shrink.convertTo(shrink, IMREAD_GRAYSCALE);
    cv::imshow("shrink", shrink);

    waitKey(0);

    return 0;
}