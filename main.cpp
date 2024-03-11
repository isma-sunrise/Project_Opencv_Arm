#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv )
{
    // read image
    cv::Mat  image_src = cv::imread( "oldcar.jpg" , cv::IMREAD_COLOR );

    // show image
    cv::imshow("image_src", image_src);
    return 0;
}
