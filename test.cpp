#include<opencv2/imgcodecs.hpp>  // 图像读写头文件 
#include<opencv2/highgui.hpp>    // 窗口显示头文件 
#include<opencv2/opencv.hpp>     // 基本函数头文件 
#include<opencv2/imgproc.hpp>    // 图像处理头文件 
#include<opencv2/calib3d.hpp>    // PnP 相关头文件 
#include<iostream> 
 
using namespace std;
using namespace cv;
 
struct LightBar {
    RotatedRect rect;
    float width;
    float height;
    Point2f center;
    float angle;
};
 
struct ArmorPlate {
    Point2f center;
    Size2f size;
    float angle;
    // 新增：存储 PnP 计算结果 
    Mat rvec;
    Mat tvec;
};
 
int a = 100;
 
Mat resolution(Mat img) {
    // 1：将图像转换为灰度图 
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    // 计算 HSV 的均值和标准差 
    Scalar mean, stddev;
    meanStdDev(img, mean, stddev);
 
    // 自动设置 HSV 范围（示例： 的容差） 
    int h_min = max(0, (int)(mean[0] - a));
    int h_max = min(180, (int)(mean[0] + a));
    int s_min = max(0, (int)(mean[1] - a));
    int s_max = min(255, (int)(mean[1] + a));
    int v_min = max(0, (int)(mean[2] - a));
    int v_max = min(255, (int)(mean[2] + a));
    Mat mask;
    inRange(img, cv::Scalar(h_min, s_min, v_min), cv::Scalar(h_max, s_max, v_max), mask);
    bitwise_not(mask, mask);
 
    // 形态学操作，去除噪声 
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
 
    return mask;
}
 
vector<LightBar> findLightBar(Mat& mask, Mat& img) {
    vector<LightBar> lightBars;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Rect> boundRect;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
 
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 100) continue;
        RotatedRect rect = minAreaRect(contours[i]);
 
        float ratio = (float)max(rect.size.height, rect.size.width) / min(rect.size.height, rect.size.width);
        if (ratio > 1.2 && ratio < 8) {
            LightBar lightBar;
            lightBar.rect = rect;
            lightBar.width = rect.size.width;
            lightBar.height = rect.size.height;
            lightBar.center = rect.center;
            lightBar.angle = rect.angle;
            lightBars.push_back(lightBar);
            // 绘制灯条矩形 
            Point2f vertices[4];
            rect.points(vertices);  // 获取旋转矩形的四个顶点 
            for (int j = 0; j < 4; j++) { // 绘制旋转矩形的边 
                line(img, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
            }
        }
    }
    return lightBars;
}
 
Mat paring_light(vector<LightBar>& lightBars, Mat& img, Mat cameraMatrix, Mat distCoeffs) {
    vector<ArmorPlate> armorPlates;
    sort(lightBars.begin(), lightBars.end(), [](const LightBar& a, const LightBar& b) {
        return a.rect.center.x < b.rect.center.x;
        });
 
    for (size_t i = 0; i < lightBars.size(); i++) {
        for (size_t j = i + 1; j < lightBars.size(); j++) {
            LightBar& left = lightBars[i];
            LightBar& right = lightBars[j];
            float angle = abs(left.angle - right.angle);
            if (angle > 90) angle = 180 - angle;
            float distance = sqrt(pow(left.center.x - right.center.x, 2) + pow(left.center.y - right.center.y, 2));
            float heightDiff = abs(left.height - right.height);
            float widthDiff = abs(left.width - right.width);
            float yDiff = abs(left.center.y - right.center.y);
            float heightAvg = (left.height + right.height) / 2;
            float widthAvg = (left.width + right.width) / 2;
            bool condition1 = angle < 15;                  // 角度差 
            bool condition2 = yDiff < heightAvg * 0.5;     // y 方向对齐 
            bool condition3 = distance > left.width * 1.2 &&  // 间距合理 
                distance < left.width * 5.0;
            bool condition4 = heightDiff < heightAvg * 0.3;   // 高度相似 
 
            if (condition1 && condition2 && condition3 && condition4) {
                // 计算装甲板参数 
                ArmorPlate armor;
                armor.center = (left.center + right.center) / 2;
                armor.size = Size2f(widthAvg, distance);
                armor.angle = (left.angle + right.angle) / 2;
 
                armorPlates.push_back(armor);
 
                // 绘制装甲板 
                RotatedRect armorRect(armor.center, armor.size, armor.angle);
                Point2f vertices[4];
                armorRect.points(vertices);
                for (int k = 0; k < 4; k++) {
                    line(img, vertices[k], vertices[(k + 1) % 4], Scalar(0, 0, 255), 3);
                }
 
                // 定义装甲板的三维模型（单位：米） 
                vector<Point3f> objectPoints;
                objectPoints.push_back(Point3f(-armor.size.width / 2, -armor.size.height / 2, 0));
                objectPoints.push_back(Point3f(armor.size.width / 2, -armor.size.height / 2, 0));
                objectPoints.push_back(Point3f(armor.size.width / 2, armor.size.height / 2, 0));
                objectPoints.push_back(Point3f(-armor.size.width / 2, armor.size.height / 2, 0));
 
                // 获取装甲板的二维图像点 
                vector<Point2f> imagePoints;
                imagePoints.push_back(vertices[0]);
                imagePoints.push_back(vertices[1]);
                imagePoints.push_back(vertices[2]);
                imagePoints.push_back(vertices[3]);
 
                // 进行 PnP 计算 
                solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, armor.rvec, armor.tvec);
 
                // 显示 PnP 坐标 
                stringstream ss;
                ss << "X: " << armor.tvec.at<double>(0) << " Y: " << armor.tvec.at<double>(1) << " Z: " << armor.tvec.at<double>(2);
                putText(img, ss.str(), armor.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
            }
        }
    }
 
    return img;
}
 
int main() {
    VideoCapture cap(0);
    int EV = 8;
    namedWindow("Exposure", WINDOW_AUTOSIZE);
    createTrackbar("EV", "Exposure", &EV, 10);
 
    // 相机内参和畸变系数（需要根据实际情况进行校准） 
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
 
    while (true) {
        cap.set(CAP_PROP_EXPOSURE, -EV);
        Mat img;
        cap >> img;
        if (img.empty())  break;
 
        Mat mask = resolution(img);
        vector<LightBar> lightBars = findLightBar(mask, img);
        img = paring_light(lightBars, img, cameraMatrix, distCoeffs);
 
        namedWindow("Video", WINDOW_AUTOSIZE);
        namedWindow("mask", WINDOW_AUTOSIZE);
        imshow("mask", mask);
        imshow("Video", img);
 
        if (waitKey(1) == 27) break; // 按 ESC 键退出 
    }
 
    cap.release();
    destroyAllWindows();
    return 0;
}