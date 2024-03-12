#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv )
{
    // read image
    cv::Mat  image_src = cv::imread( "../oldcar.jpg" , cv::IMREAD_COLOR );
    // show image
        /* SHOW IMAGES */
    if ( !image_src.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::imshow("image_src", image_src);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
