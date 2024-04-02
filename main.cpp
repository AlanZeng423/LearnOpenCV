#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
int main() {
    cout<<"1"<<endl;
    Mat image = imread("../1.png");
    if (image.empty()){
        printf("Image not loaded");
        return -1;
    }
    imshow("image", image);
    waitKey(0);
    return 0;
}