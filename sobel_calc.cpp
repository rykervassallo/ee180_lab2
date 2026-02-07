#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  double color;

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j++) {
      color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
    }
  }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
// void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
// {
//   Mat img_outx = img_gray.clone();
//   Mat img_outy = img_gray.clone();

//   // Apply Sobel filter to black & white image
//   unsigned short sobel;

//   // Calculate the x convolution
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//       sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
// 		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
// 		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
// 		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//       sobel = (sobel > 255) ? 255 : sobel;
//       img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
//     }
//   }

//   // Calc the y convolution
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//      sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//      sobel = (sobel > 255) ? 255 : sobel;

//      img_outy.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }

//   // Combine the two convolutions into the output image
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//       sobel = img_outx.data[IMG_WIDTH*(i) + j] +
// 	img_outy.data[IMG_WIDTH*(i) + j];
//       sobel = (sobel > 255) ? 255 : sobel;
//       img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }
// }

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
// void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
// {
//   // Apply Sobel filter to black & white image
//   unsigned short sobel;
//   unsigned short sobel_x;
//   unsigned short sobel_y;

//   // Calculate x and y convolutions and combine in single loop
//   for (int i=1; i<img_gray.rows-1; i++) {
//     for (int j=1; j<img_gray.cols-1; j++) {
//       // Calculate the x convolution

//       sobel_x = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
// 		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
// 		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
// 		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//       // Calc the y convolution
//       sobel_y = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//       // Combine the two convolutions
//       sobel = sobel_x + sobel_y;
//       sobel = (sobel > 255) ? 255 : sobel;
//       img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }
// }

// void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
// {
//   // Apply Sobel filter to black & white image
//   unsigned short sobel;
//   unsigned short sobel_x;
//   unsigned short sobel_y;

//   // Calculate x and y convolutions and combine in single loop
//   for (int i=1; i<img_gray.rows-1; i++) {
//     for (int j=1; j<img_gray.cols-1; j++) {
//       // Cache the 9 pixel values needed for the 3x3 kernel
//       unsigned char p00 = img_gray.data[IMG_WIDTH*(i-1) + (j-1)];
//       unsigned char p01 = img_gray.data[IMG_WIDTH*(i-1) + (j)];
//       unsigned char p02 = img_gray.data[IMG_WIDTH*(i-1) + (j+1)];
      
//       unsigned char p10 = img_gray.data[IMG_WIDTH*(i) + (j-1)];
//       unsigned char p12 = img_gray.data[IMG_WIDTH*(i) + (j+1)];
      
//       unsigned char p20 = img_gray.data[IMG_WIDTH*(i+1) + (j-1)];
//       unsigned char p21 = img_gray.data[IMG_WIDTH*(i+1) + (j)];
//       unsigned char p22 = img_gray.data[IMG_WIDTH*(i+1) + (j+1)];

//       // Calculate the x convolution using cached values
//       sobel_x = abs(p00 - p20 + 2*p01 - 2*p21 + p02 - p22);

//       // Calc the y convolution using cached values
//       sobel_y = abs(p00 - p02 + 2*p10 - 2*p12 + p20 - p22);

//       // Combine the two convolutions
//       sobel = sobel_x + sobel_y;
//       sobel = (sobel > 255) ? 255 : sobel;
//       img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }
// }

void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  for (int i = 1; i < img_gray.rows - 1; i++) {
    // Pre-calculate row pointers
    unsigned char* row_prev = img_gray.data + IMG_WIDTH * (i - 1);
    unsigned char* row_curr = img_gray.data + IMG_WIDTH * i;
    unsigned char* row_next = img_gray.data + IMG_WIDTH * (i + 1);
    unsigned char* out_row = img_sobel_out.data + IMG_WIDTH * i;
    
    // Initialize sliding window for first pixel
    unsigned char p00 = row_prev[0], p01 = row_prev[1], p02 = row_prev[2];
    unsigned char p10 = row_curr[0], p12 = row_curr[2];
    unsigned char p20 = row_next[0], p21 = row_next[1], p22 = row_next[2];
    
    for (int j = 1; j < img_gray.cols - 1; j++) {
      // Calculate convolutions
      int sobel_x = abs(p00 - p20 + 2*p01 - 2*p21 + p02 - p22);
      int sobel_y = abs(p00 - p02 + 2*p10 - 2*p12 + p20 - p22);
      
      int sobel = sobel_x + sobel_y;
      out_row[j] = (sobel > 255) ? 255 : sobel;
      
      // Slide the window right (reuse 6 values, load only 3 new ones)
      p00 = p01; p01 = p02; p02 = row_prev[j + 2];
      p10 = p12; p12 = row_curr[j + 2];
      p20 = p21; p21 = p22; p22 = row_next[j + 2];
    }
  }
}