//
// Created by Alan on 2024/3/22.
//
#include <opencv2/opencv.hpp>
#include <random>

using namespace cv;
using namespace std;

// 函数添加高斯噪声
void addGaussianNoise(Mat &image) {
    Mat noise(image.size(), image.type());
    double mean = 0.0;
    double stddev = 30.0;  // 标准差
    randn(noise, mean, stddev); // 生成正态分布随机数
    image += noise;  // 添加噪声
}

// 函数添加椒盐噪声
void addSaltAndPepperNoise(Mat &image, double saltPerc, double pepperPerc) {
    int rows = image.rows;
    int cols = image.cols;
    int ch = image.channels();
    int num_salt = static_cast<int>((rows * cols * ch) * saltPerc);
    int num_pepper = static_cast<int>((rows * cols * ch) * pepperPerc);

    for (int i = 0; i < num_salt; i++) {
        int r = rand() % rows;
        int c = rand() % cols;
        int ch = rand() % 3;
        image.at<Vec3b>(r, c)[ch] = 255;
    }

    for (int i = 0; i < num_pepper; i++) {
        int r = rand() % rows;
        int c = rand() % cols;
        int ch = rand() % 3;
        image.at<Vec3b>(r, c)[ch] = 0;
    }
}

// TODO: OpenCV实现高斯滤波
void GaussianBlurByOpenCV(const Mat &image, Mat &result, int size=5, double sigma=5) {
    GaussianBlur(image, result, Size(size, size), sigma, 0);
}

// 生成高斯滤波核
Mat generateGaussKernel(int size, double sigma){
    Size wsize(size, size);
    Mat Kernel = Mat(wsize,CV_64F);
    int center = (size - 1) / 2;
    double sum = 0.0;
    double x, y;
    for (int i = 0; i < size; ++i){
        y = pow(i - center, 2);
        for (int j = 0; j < size; ++j){
            x = pow(j - center, 2);
            //因为最后都要归一化的，常数部分可以不计算，也减少了运算量
            double g = exp(-(x + y) / (2 * sigma*sigma));
            Kernel.at<double>(i, j) = g;
            sum += g;
        }
    }
    Kernel = Kernel / sum;
    return Kernel;
}

// TODO: 实现高斯滤波
void GaussianFilter(const Mat& src, Mat& dst, int size=5, double sigma=5) {
    Mat Kernel;
    Kernel = generateGaussKernel(size, sigma);
    int border = size / 2;
    dst = Mat::zeros(src.size(), src.type());
    //边界填充
    Mat newSrc;
    copyMakeBorder(src, newSrc, border, border, border, border, BORDER_REFLECT);

    for (int i = border; i < src.rows + border; ++i) { // 外层循环遍历原图像
        for (int j = border; j < src.cols + border; ++j) { // 外层循环遍历原图像
            double sum = 0.0;
            for (int m = 0; m < size; ++m) { // 内层循环遍历卷积核
                for (int n = 0; n < size; ++n) { // 内层循环遍历卷积核
                    // 卷积操作
                    sum += newSrc.at<uchar>(i + m - border, j + n - border) * Kernel.at<double>(m, n);
                }
            }
            dst.at<uchar>(i - border, j - border) = sum;
        }
    }
}

// TODO: OpenCV实现中值滤波
void MedianBlurByOpenCV(const Mat &src, Mat &dst, int size=5) {
    medianBlur(src, dst, size);
}

// TODO: 实现中值滤波
void MedianFilter(const Mat &src, Mat &dst, int size=5) {
    int border = size / 2;
    dst = Mat::zeros(src.size(), src.type());
    //边界填充
    Mat newSrc;
    copyMakeBorder(src, newSrc, border, border, border, border, BORDER_REFLECT);

    for (int i = border; i < src.rows + border; ++i) { // 外层循环遍历原图像
        for (int j = border; j < src.cols + border; ++j) { // 外层循环遍历原图像
           vector<uchar> vec;
            for (int m = 0; m < size; ++m) { // 内层循环遍历卷积核
                for (int n = 0; n < size; ++n) { // 内层循环遍历卷积核
                    // 计算中值
                    vec.push_back(newSrc.at<uchar>(i + m - border, j + n - border));
                }
            }
           sort(vec.begin(), vec.end());
            dst.at<uchar>(i - border, j - border) = vec[vec.size() / 2];
        }
    }
}


int main() {
    // 读取图像
    Mat image = imread("../TJU.jpg");
    if (image.empty()) {
       cerr << "Image load failed" <<endl;
        return -1;
    }
    // 原图为彩色, 灰化
    cvtColor(image, image, COLOR_BGR2GRAY);
    imwrite("../Assign1/TJU_Origin.jpg", image);

    // 添加高斯噪声
    Mat imageWithGaussianNoise = image.clone();
    addGaussianNoise(imageWithGaussianNoise);
    imwrite("../Assign1/TJU_GaussianNoise.jpg", imageWithGaussianNoise);

    // 添加椒盐噪声
    Mat imageWithSaltAndPepperNoise = image.clone();
    addSaltAndPepperNoise(imageWithSaltAndPepperNoise, 0.05, 0.05);
    imwrite("../Assign1/TJU_SaltAndPepperNoise.jpg", imageWithSaltAndPepperNoise);


    // 高斯滤波: 对加有高斯噪声的图像进行高斯滤波
    Mat imageGaussianBlurByOpenCV;
    Mat imageGaussianBlurBySelf;
    Mat imageGaussianBlur = imageWithGaussianNoise.clone();
    GaussianBlurByOpenCV(imageGaussianBlur, imageGaussianBlurByOpenCV, 5, 0.8);
    GaussianFilter(imageGaussianBlur, imageGaussianBlurBySelf, 5, 0.8);
    imshow("Gaussian Blur with Gaussian Noise", imageGaussianBlurByOpenCV);
    imshow("Gaussian Blur with Gaussian Noise by Self", imageGaussianBlurBySelf);
    imwrite("../Assign1/TJU_GaussianBlurByOpenCV.jpg", imageGaussianBlurByOpenCV);
    imwrite("../Assign1/TJU_GaussianBlurBySelf.jpg", imageGaussianBlurBySelf);


    // 中值滤波: 对加有椒盐噪声的图像进行中值滤波
    Mat imageMedianBlurByOpenCV;
    Mat imageMedianBlurBySelf;
    Mat imageMedianBlur = imageWithSaltAndPepperNoise.clone();
    MedianBlurByOpenCV(imageMedianBlur, imageMedianBlurByOpenCV, 5);
    MedianFilter(imageMedianBlur, imageMedianBlurBySelf, 5);
    imshow("Median Blur with Salt and Pepper Noise by OpenCV", imageMedianBlurByOpenCV);
    imshow("Median Blur with Salt and Pepper Noise by Self", imageMedianBlurBySelf);
    imwrite("../Assign1/TJU_MedianBlurByOpenCV.jpg", imageMedianBlurByOpenCV);
    imwrite("../Assign1/TJU_MedianBlurBySelf.jpg", imageMedianBlurBySelf);


    waitKey(0); // 等待按键

    return 0;
}
