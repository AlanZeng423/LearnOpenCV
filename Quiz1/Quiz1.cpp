//
// Created by Alan on 2024/3/26.
//

/* Quiz
- 绘制光流
- 加载视频。
- 调用库函数寻找特征点。
- 调用库函数计算两帧图像中特征点的移动距离。
- 删除未移动的特征点。
- 在移动的点之间绘制线段。
使用openCV库和c++实现
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // 加载视频
    VideoCapture cap("../cars.mp4");

    // 检测视频是否成功打开
    if (!cap.isOpened()) {
        cerr << "Error opening video file\n";
        return -1;
    }

    // 创建两个Mat对象，用于存储两帧图像
    Mat old_frame, old_gray;
    // 创建两个vector对象，用于存储特征点
    vector<Point2f> p0, p1; // Point2f: 2D point with floating-point coordinates

    // 拿取第一帧图像, 并转换为灰度图
    cap >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    // 使用`goodFeaturesToTrack`从第一帧图像中提取特征点
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // 获取视频的属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)); // 宽
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); // 高
    double fps = cap.get(cv::CAP_PROP_FPS); // 帧率

    // 创建视频写入对象, 用于存储最后的结果
    cv::VideoWriter videoWriter("../Quiz1/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(frame_width, frame_height), true);

    // 检测视频写入对象是否成功打开
    if (!videoWriter.isOpened()) {
        std::cerr << "Error opening video writer" << std::endl;
        return -1;
    }

    // 创建一个mask图像, 用于绘制
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    while(true) { // 循环读取视频帧
        Mat frame, frame_gray; // 创建两个Mat对象, 用于存储当前帧图像和灰度图
        cap >> frame; // 读取视频帧
        if (frame.empty()) { // 如果视频帧为空, 退出循环
            break;
        }
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY); // 将当前帧图像转换为灰度图


        // 计算光流
        vector<uchar> status; // 创建一个vector对象, 用于存储特征点的状态
        vector<float> err; // 创建一个vector对象, 用于存储特征点的误差

        // 使用`calcOpticalFlowPyrLK`, 计算两帧图像中特征点的移动距离
        // status: 0表示特征点i没有移动, 1表示特征点i移动了
        // err: 特征点i的误差
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);


        // 遍历特征点, 选择好的特征点, 并绘制线段
        for (uint i = 0; i < p0.size(); i++) {

            // Select good points
            if (status[i] == 0) { // 如果status[i] == 0, 表示特征点i没有移动
                // 如果特征点i没有移动, 删除特征点i
                p0.erase(p0.begin() + i);
                p1.erase(p1.begin() + i);
                continue;
            }
            // 如果特征点i移动了, 绘制线段
            line(mask, p1[i], p0[i], Scalar(0, 255, 0), 2);
            // line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
            // thickness: 线的宽度

            // 绘制特征点
            circle(frame, p1[i], 5, Scalar(0, 255, 0), -1);
            // circle(img, center, radius, color[, thickness[, lineType[, shift]]])}
            // thickness = -1: 表示填充圆
        }


        Mat img; // 创建一个Mat对象, 用于存储最终结果
        add(frame, mask, img); // 将frame和mask相加, 存储到img中
        videoWriter.write(img); // 将img写入到视频中

        imshow("Frame", img); // 显示每一帧的img, 最终效果为视频
        int keyboard = waitKey(30); // 等待30ms, 如果按下键盘, 则退出循环
        if (keyboard == 'q' || keyboard == 27)
            break;

        // 更新old_gray和p0
        old_gray = frame_gray.clone(); // 更新old_gray
        p0 = p1; // 更新p0,

    }
    // 释放资源
    videoWriter.release();
    cap.release();
    destroyAllWindows();
    return 0;
}