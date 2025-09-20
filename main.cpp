#include <cstddef>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main()
{
    // get image
    // Mat image = imread("/home/angui/OpenCV_Project/resources/test_image.png");

    // // 检查图像是否成功加载
    // if (image.empty()) {
    //     std::cerr << "Error: Could not load image!" << std::endl;
    //     return -1;
    // }
    
    // // 打印图像信息
    // cout << "Image channels: " << image.channels() << endl;
    // cout << "Image size: " << image.rows << "x" << image.cols << endl;
    // // turn to Grayscale
    // Mat gray_image;
    // cvtColor(image, gray_image, COLOR_BGR2GRAY);
    // // show image
    // imshow("Gray Image", gray_image);
    // // turn to hsv
    // Mat hsv_image;
    // cvtColor(image, hsv_image, COLOR_BGR2HSV);
    // imshow("HSV Image", hsv_image);

    // //apply some filter
    // // blur filter
    // Mat blur_image;
    // blur(gray_image, blur_image, Size(5, 5));
    // // show filtered image
    // imshow("Blur Filtered Image", blur_image);
    
    // // gauss filter
    // Mat gaussian_image;
    // GaussianBlur(gray_image, gaussian_image, Size(5, 5), 0);
    // // show gaussian filtered image
    // imshow("Gaussian Filtered Image", gaussian_image);

    // // get characteristics
    // // get red area
    // Mat red_area;
    // // inRange(image, Scalar(0, 0, 100), Scalar(100, 100, 255), red_area);
    // // show red area, use hsv way
    // inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), red_area);
    // imshow("Red Area", red_area);
    // // search red outedge in this image
    // Mat hsv, hsvChannels[3];
    // cvtColor(image, hsv, COLOR_BGR2HSV);
    // split(hsv, hsvChannels);
    // Mat red_edges;
    // Canny(hsvChannels[0], red_edges, 100, 200);
    // imshow("color edge (H channel)", red_edges);

    // Mat morph;
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // morphologyEx(red_edges, morph, MORPH_CLOSE, kernel);
    // imshow("morphologyEx", morph);
    // // search red bounding box in this image
    // vector<Vec4i> hierarchy;
    // vector<vector<Point>> contours;
    // findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Mat contourImage = Mat::zeros(morph.size(), CV_8UC3);
    // for (size_t i = 0; i < contours.size(); i++) {
    //     cv::drawContours(contourImage, contours, i, cv::Scalar(0, 0, 255), 2);
    // }
    // imshow("Contours", contourImage);
    // for (size_t i = 0; i < contours.size(); i++) {
    //     double area = cv::contourArea(contours[i]);
    //     if (area < 500) {
    //         continue;
    //     }
    //     cv::Rect bounding_box = cv::boundingRect(contours[i]);
    //     cv::rectangle(contourImage, bounding_box, cv::Scalar(0, 0, 255), 2);
    // }
    // imshow("Contours with Bounding Boxes", contourImage);
    
    // // 只有在轮廓存在时才计算面积
    // double red_edge_area = 0;
    // if (!contours.empty()) {
    //     red_edge_area = cv::contourArea(contours[0]);
    // }
    // cout << "Red edge area: " << red_edge_area << endl;

    // // get the lightcolor area and process
    // // get the lightcolor area
    // cv::Mat lightcolor_area;
    // cv::inRange(image, cv::Scalar(100, 100, 100), cv::Scalar(255, 255, 255), lightcolor_area);
    // // graify - 删除这行，因为lightcolor_area已经是单通道了
    // // cv::Mat lightcolor_gray;
    // // cv::cvtColor(lightcolor_area, lightcolor_gray, cv::COLOR_BGR2GRAY);
    
    // // 直接使用lightcolor_area作为灰度图像
    // cv::Mat lightcolor_binary;
    // cv::threshold(lightcolor_area, lightcolor_binary, 128, 255, cv::THRESH_BINARY);
    
    // // inflate
    // cv::Mat lightcolor_inflate;
    // cv::dilate(lightcolor_binary, lightcolor_inflate, cv::Mat(), cv::Point(-1, -1), 2);
    // // corrode
    // cv::Mat lightcolor_corrode;
    // cv::erode(lightcolor_inflate, lightcolor_corrode, cv::Mat(), cv::Point(-1, -1), 2);
    // // flood fill
    // cv::Mat lightcolor_flood;
    // lightcolor_corrode.copyTo(lightcolor_flood);
    
    // // 检查坐标点是否在图像范围内
    // if (image.rows > 0 && image.cols > 0) {
    //     cv::floodFill(lightcolor_flood, cv::Point(0, 0), cv::Scalar(255));
    // }
    // cv::imshow("Lightcolor Flood", lightcolor_flood);

    // // plot
    // // plot a circle, a rectangle and a word
    // Mat image2 = image.clone();
    // cv::circle(image2, cv::Point(100, 100), 50, cv::Scalar(0, 0, 255), 2);
    // cv::rectangle(image2, cv::Point(200, 200), cv::Point(300, 300), cv::Scalar(0, 255, 0), 2);
    // cv::putText(image2, "OpenCV", cv::Point(100, 400), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    // cv::imshow("Plot", image2);

    // // process image
    // // rolling 35 degree
    // cv::Mat rotated_image;
    // cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point(image.cols / 2, image.rows / 2), 35, 1.0);
    // cv::warpAffine(image, rotated_image, rotation_matrix, cv::Size(image.cols, image.rows));
    // cv::imshow("Rotated Image", rotated_image);
    
    // // cut origin image on the 1/4 leftup
    // // 检查图像尺寸是否足够大
    // if (image.rows >= 2 && image.cols >= 2) {
    //     cv::Mat cut_image = image(cv::Rect(0, 0, image.cols / 2, image.rows / 2));
    //     cv::imshow("Cut Image", cut_image);
    // }

    Mat img = imread("./resources/test_image_2.png");
    Mat gray, bright;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // get bright area
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(gray, &minVal, &maxVal, &minLoc, &maxLoc);
    threshold(gray, bright, maxVal * 0.95, 255, THRESH_BINARY);
    imshow("Bright", bright);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(bright, bright, MORPH_CLOSE, kernel);
    imshow("Bright Close", bright);
    vector<vector<Point>> contours;
    findContours(bright, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    imshow("Bright Contours", bright);
    vector<RotatedRect> stripes;
    for (auto& c : contours){
        RotatedRect rect = minAreaRect(c);
        float w = rect.size.width, h = rect.size.height;
        if (max(w,h) > 100 && min(w, h) < 20){
            stripes.push_back(rect);
        }
    }

    for (size_t i = 0; i < stripes.size(); i++){
        for (size_t j = i + 1; j < stripes.size(); ++j){
            float angleDiff = abs(stripes[i].angle - stripes[j].angle);
            float centerDist = norm(stripes[i].center - stripes[j].center);
            if (angleDiff < 10 && centerDist < 20000){
                Point2f pts1[4], pts2[4];
                stripes[i].points(pts1);
                stripes[j].points(pts2);
                for (int k = 0; k < 4; k++){
                    line(img, pts1[k], pts2[(k+1) % 4], Scalar(255, 0, 0), 2);
                    line(img, pts1[k], pts2[(k+1) % 4], Scalar(255, 0, 0), 2);
                }
            }
        }
    }
    imshow("Stripes", img);
    waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}