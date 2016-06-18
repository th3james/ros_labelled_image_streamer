#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>

#include "dir_utils.hpp"

// Compile with clang++ code.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc

std::string IMAGE_DIR = "../training-data/rects/present";

int main() {
  typedef std::vector<std::string> str_vec_t;

  str_vec_t images(DirUtils::list_directory(IMAGE_DIR));

  for (str_vec_t::const_iterator iter = images.begin(); iter != images.end(); iter++) {
    std::cout << "rotating " << *iter << std::endl;
    cv::Mat src = cv::imread(*iter, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat dst;

    cv::Point2f pc(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, -5, 1.2);

    cv::warpAffine(src, dst, r, src.size());

    cv::imwrite(*iter + "-rotated-left.jpg", dst);

    r = cv::getRotationMatrix2D(pc, 5, 1.2);

    cv::warpAffine(src, dst, r, src.size());

    cv::imwrite(*iter + "-rotated-right.jpg", dst);
  }

  return 0;
}
