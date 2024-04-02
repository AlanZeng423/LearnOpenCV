//
// Created by Alan on 2024/4/2.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int main() {
    // 打开视频文件
//    string videoPath = "../cars.mp4";
//    string videoPath = "../234_1712047332.mp4";
    string videoPath = "../ball.mp4";
//    string videoPath = "../鸡你太美.mp4";
    // 首先检查文件是否存在
    ifstream file(videoPath);
    if (!file.good()) {
        cerr << "Error: File not found or cannot be opened: " << videoPath << endl;
        return -1;
    }

    // 打开视频文件
    VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        cerr << "Error opening video file: " << videoPath << endl;
        return -1;
    }

    // 读取第一帧
    Mat frame;
    capture >> frame;
    if (frame.empty()) {
        cerr << "Error reading video" << endl;
        return -1;
    }

    imshow("Frame", frame);
    waitKey(1);  // 短暂等待确保图像显示

    // 选择跟踪目标
    Rect track_window = selectROI("Tracking", frame, true, false);
    if (track_window.width <= 0 || track_window.height <= 0) {
        cerr << "Error: Track window has invalid size." << endl;
        cerr << "track_window.width = " << track_window.width << endl;
        cerr << "track_window.height = " << track_window.height << endl;
        cerr << "track_window.size() = " << track_window.size() << endl;
        cout << "Selected ROI: " << track_window << endl;
        return -1;
    } else {
        cout << "Selected ROI: " << track_window << endl;
    }


    // 设置ROI（感兴趣区域）用于跟踪
    Mat roi = frame(track_window);
    if (roi.empty()) {
        cerr << "Error: ROI is empty." << endl;
        return -1;
    }

    // 将ROI转换为HSV颜色空间

    Mat hsv_roi, mask;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);





    inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

//    // 计算ROI的直方图
//    Mat roi_hist;
//    int histSize = 180;
//    float range[] = { 0, 180 };
//    const float* histRange = { range };
//    calcHist(&hsv_roi, 1, 0, mask, roi_hist, 1, &histSize, &histRange);
//    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    // 分离HSV通道
    vector<Mat> hsv_channels;
    split(hsv_roi, hsv_channels);
    Mat hue_channel = hsv_channels[0]; // 取H通道

    cout << "hue_channel.size() = " << hue_channel.size() << endl;
    // 输出hue_channel的通道数
    cout << "Hue channel type: " << hue_channel.type() << endl;
    cout << "Hue channel channels: " << hue_channel.channels() << endl;

    // 计算H通道的直方图
    Mat roi_hist;
    int histSize = 180;
    float range[] = { 0, 180 };
    const float* histRange = { range };
    calcHist(&hue_channel, 1, 0, mask, roi_hist, 1, &histSize, &histRange,true, false);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    // 设置追踪参数
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);

    // 开始处理视频
    while (true) {
        Mat hsv, dst;
        capture >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);

        vector<Mat> hsv_cs;
        split(hsv, hsv_cs);
        Mat hue_c = hsv_cs[0]; // 取H通道


        calcBackProject(&hue_c, 1, 0, roi_hist, dst, &histRange);

        // 使用MeanShift算法跟踪
        meanShift(dst, track_window, term_crit);

        // 画出它
        rectangle(frame, track_window, Scalar(0, 255, 0), 2);

        // 使用CamShift算法跟踪
        RotatedRect track_box = CamShift(dst, track_window, term_crit);

        // 画出它
        Point2f points[4];
        track_box.points(points);
        for (int i = 0; i < 4; i++)
            line(frame, points[i], points[(i+1)%4], Scalar(0, 0, 255), 2);

        imshow("Tracking", frame);

        // 按'q'退出
        if (waitKey(30) == 'q') break;
    }

    capture.release();
    destroyAllWindows();
    return 0;
}
