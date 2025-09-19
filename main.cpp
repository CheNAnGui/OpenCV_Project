#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    // get image
    cv::Mat image = cv::imread("/home/angui/OpenCV_Project/resources/test_image.png");
    
    // 检查图像是否成功加载
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    
    // 打印图像信息
    std::cout << "Image channels: " << image.channels() << std::endl;
    std::cout << "Image size: " << image.rows << "x" << image.cols << std::endl;
    // turn to Grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    // show image
    cv::imshow("Gray Image", gray_image);

    // m filter
    cv::Mat filtered_image;
    cv::medianBlur(gray_image, filtered_image, 5);
    // show filtered image
    cv::imshow("Median Filtered Image", filtered_image);
    
    // gauss filter
    cv::Mat gaussian_image;
    cv::GaussianBlur(gray_image, gaussian_image, cv::Size(5, 5), 0);
    // show gaussian filtered image
    cv::imshow("Gaussian Filtered Image", gaussian_image);

    // get characteristics
    // get red area
    cv::Mat red_area;
    cv::inRange(image, cv::Scalar(0, 0, 100), cv::Scalar(100, 100, 255), red_area);
    // show red area, use hsv way
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red_area);
    cv::imshow("Red Area", red_area);
    // search red outedge in this image
    cv::Mat red_edges;
    cv::Canny(red_area, red_edges, 100, 200);
    cv::imshow("Red Edges", red_edges);
    // search red bounding box in this image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 检查是否有找到轮廓
    if (contours.empty()) {
        std::cerr << "Error: No contours found!" << std::endl;
        return -1;
    }
    
    // draw red bounding box
    for (const auto& contour : contours) {
        cv::Rect bounding_box = cv::boundingRect(contour);
        cv::rectangle(image, bounding_box, cv::Scalar(0, 255, 0), 2);
    }
    
    // 只有在轮廓存在时才计算面积
    double red_edge_area = 0;
    if (!contours.empty()) {
        red_edge_area = cv::contourArea(contours[0]);
    }
    std::cout << "Red edge area: " << red_edge_area << std::endl;

    // get the lightcolor area and process
    // get the lightcolor area
    cv::Mat lightcolor_area;
    cv::inRange(image, cv::Scalar(100, 100, 100), cv::Scalar(255, 255, 255), lightcolor_area);
    // graify - 删除这行，因为lightcolor_area已经是单通道了
    // cv::Mat lightcolor_gray;
    // cv::cvtColor(lightcolor_area, lightcolor_gray, cv::COLOR_BGR2GRAY);
    
    // 直接使用lightcolor_area作为灰度图像
    cv::Mat lightcolor_binary;
    cv::threshold(lightcolor_area, lightcolor_binary, 128, 255, cv::THRESH_BINARY);
    
    // inflate
    cv::Mat lightcolor_inflate;
    cv::dilate(lightcolor_binary, lightcolor_inflate, cv::Mat(), cv::Point(-1, -1), 2);
    // corrode
    cv::Mat lightcolor_corrode;
    cv::erode(lightcolor_inflate, lightcolor_corrode, cv::Mat(), cv::Point(-1, -1), 2);
    // flood fill
    cv::Mat lightcolor_flood;
    lightcolor_corrode.copyTo(lightcolor_flood);
    
    // 检查坐标点是否在图像范围内
    if (image.rows > 0 && image.cols > 0) {
        cv::floodFill(lightcolor_flood, cv::Point(0, 0), cv::Scalar(255));
    }
    cv::imshow("Lightcolor Flood", lightcolor_flood);

    // plot
    // plot a circle, a rectangle and a word
    cv::circle(image, cv::Point(100, 100), 50, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(image, cv::Point(200, 200), cv::Point(300, 300), cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "OpenCV", cv::Point(100, 400), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::imshow("Plot", image);
    // plot a red outline
    cv::Mat red_outline;
    cv::cvtColor(image, red_outline, cv::COLOR_BGR2GRAY);
    cv::Canny(red_outline, red_outline, 100, 200);
    cv::imshow("Red Outline", red_outline);
    
    // 只有在轮廓存在时才绘制边界框
    if (!contours.empty()) {
        cv::Rect red_bounding_box = cv::boundingRect(contours[0]);
        cv::rectangle(image, red_bounding_box, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Red Bounding Box", image);
    }

    // process image
    // rolling 35 degree
    cv::Mat rotated_image;
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point(image.cols / 2, image.rows / 2), 35, 1.0);
    cv::warpAffine(image, rotated_image, rotation_matrix, cv::Size(image.cols, image.rows));
    cv::imshow("Rotated Image", rotated_image);
    
    // cut origin image on the 1/4 leftup
    // 检查图像尺寸是否足够大
    if (image.rows >= 2 && image.cols >= 2) {
        cv::Mat cut_image = image(cv::Rect(0, 0, image.cols / 2, image.rows / 2));
        cv::imshow("Cut Image", cut_image);
    }
    
    // 等待按键按下
    cv::waitKey(0);
    
    // 释放所有窗口
    cv::destroyAllWindows();
    
    return 0;
}