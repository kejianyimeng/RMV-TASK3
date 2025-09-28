#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

std::vector<cv::Point2d> extractBlueBallPoints(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return {};
    }
    
    std::vector<cv::Point2d> trajectory_points;
    cv::Mat frame, hsv_frame, mask;
    
    // 蓝色在HSV颜色空间中的范围（针对蓝色小球调整）
    // 蓝色通常对应Hue值在100-130之间（OpenCV中Hue范围是0-180）
    cv::Scalar lower_blue(60, 60, 60);   // HSV下限：H(100-110), S(150-255), V(100-255)
    cv::Scalar upper_blue(130, 255, 255);  // HSV上限
    
    // 获取视频的FPS信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 60.0; // 默认值
    std::cout << "视频FPS: " << fps << std::endl;
    
    int frame_count = 0;
    double prev_x = -1, prev_y = -1;
    
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        // 转换为HSV颜色空间
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
        
        // 颜色阈值分割 - 检测蓝色
        cv::inRange(hsv_frame, lower_blue, upper_blue, mask);
        
        // 形态学操作去噪 - 针对小球形状优化
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        
        // 先闭运算填充小球内部可能的小孔洞
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        // 再开运算去除小噪声
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Point2d current_point(-1, -1);
        bool point_found = false;
        
        if (!contours.empty()) {
            // 按面积排序，找到最大的几个轮廓
            std::sort(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) > cv::contourArea(b);
                });
            
            // 遍历前几个大轮廓，寻找圆形度高的
            for (size_t i = 0; i < std::min(contours.size(), size_t(3)); ++i) {
                double area = cv::contourArea(contours[i]);
                double perimeter = cv::arcLength(contours[i], true);
                
                // 计算圆形度 (4πA/P²)，接近1表示更接近圆形
                double circularity = 0;
                if (perimeter > 0) {
                    circularity = (4 * CV_PI * area) / (perimeter * perimeter);
                }
                
                // 面积阈值和圆形度阈值
                if (area > 50 && area < 5000 && circularity > 0.7) {
                    cv::Moments m = cv::moments(contours[i]);
                    if (m.m00 > 0) {
                        double x = m.m10 / m.m00;
                        double y = m.m01 / m.m00;
                        
                        // 简单的运动连续性检查
                        if (prev_x >= 0 && prev_y >= 0) {
                            double distance = sqrt(pow(x - prev_x, 2) + pow(y - prev_y, 2));
                            // 如果移动距离过大，可能是噪声
                            if (distance > 100) {
                                continue;
                            }
                        }
                        
                        current_point = cv::Point2d(x, y);
                        point_found = true;
                        
                        // 可视化
                        cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
                        cv::circle(frame, cv::Point(x, y), 10, cv::Scalar(0, 255, 0), 2);
                        
                        // 显示轮廓
                        cv::drawContours(frame, contours, i, cv::Scalar(0, 0, 255), 2);
                        
                        break; // 找到合适的小球就退出循环
                    }
                }
            }
        }
        
        if (point_found) {
            trajectory_points.push_back(current_point);
            prev_x = current_point.x;
            prev_y = current_point.y;
            
            // 在图像上显示帧号和坐标
            std::string info = "Frame: " + std::to_string(frame_count) + 
                              " Pos: (" + std::to_string(int(current_point.x)) + 
                              ", " + std::to_string(int(current_point.y)) + ")";
            cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        
        // 显示处理结果
        cv::imshow("Original", frame);
        cv::imshow("Mask", mask);
        
        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC退出
        if (key == 'p') cv::waitKey(0); // 暂停
        
        frame_count++;
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "共处理 " << frame_count << " 帧，检测到 " << trajectory_points.size() << " 个数据点" << std::endl;
    
    return trajectory_points;
}

// 交互式参数调整工具（可选）
void adjustColorThreshold(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return;
    
    cv::Mat frame, hsv_frame, mask;
    
    // 创建滑动条窗口
    cv::namedWindow("Threshold Adjust", cv::WINDOW_NORMAL);
    
    int h_low = 100, h_high = 130;
    int s_low = 150, s_high = 255;
    int v_low = 100, v_high = 255;
    
    cv::createTrackbar("H Low", "Threshold Adjust", &h_low, 180);
    cv::createTrackbar("H High", "Threshold Adjust", &h_high, 180);
    cv::createTrackbar("S Low", "Threshold Adjust", &s_low, 255);
    cv::createTrackbar("S High", "Threshold Adjust", &s_high, 255);
    cv::createTrackbar("V Low", "Threshold Adjust", &v_low, 255);
    cv::createTrackbar("V High", "Threshold Adjust", &v_high, 255);
    
    while (cap.read(frame)) {
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
        
        cv::Scalar lower(h_low, s_low, v_low);
        cv::Scalar upper(h_high, s_high, v_high);
        
        cv::inRange(hsv_frame, lower, upper, mask);
        
        // 显示结果
        cv::imshow("Original", frame);
        cv::imshow("Mask", mask);
        cv::imshow("Threshold Adjust", mask);
        
        char key = cv::waitKey(30);
        if (key == 27) break;
        if (key == 's') {
            std::cout << "当前阈值: lower(" << h_low << "," << s_low << "," << v_low 
                      << "), upper(" << h_high << "," << s_high << "," << v_high << ")" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
}

// 主函数
int main() {
    std::string video_path = "/home/xujiake/RM_Task/TASK3/video.mp4";
    
    // 如果需要调整阈值，先运行这个函数
    // adjustColorThreshold(video_path);
    
    // 提取数据点
    auto points = extractBlueBallPoints(video_path);
    
    if (points.size() < 4) {
        std::cerr << "提取到的数据点不足，无法进行轨迹拟合" << std::endl;
        return -1;
    }
    
    // 输出提取到的点
    std::cout << "提取到的数据点:" << std::endl;
    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << "帧 " << i << ": (" << points[i].x << ", " << points[i].y << ")" << std::endl;
    }
    
    // 这里可以添加轨迹拟合代码（使用之前提供的TrajectoryFitter类）
    
    return 0;
}