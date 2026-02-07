#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
using namespace cv;

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
// void grayScale(Mat& img, Mat& img_gray_out)
// {
//   double color;

//   // Convert to grayscale
//   for (int i=0; i<img.rows; i++) {
//     for (int j=0; j<img.cols; j++) {
//       color = .114*img.data[STEP0*i + STEP1*j] +
//               .587*img.data[STEP0*i + STEP1*j + 1] +
//               .299*img.data[STEP0*i + STEP1*j + 2];
//       img_gray_out.data[IMG_WIDTH*i + j] = color;
//     }
//   }
// }

void grayScale(Mat& img, Mat& img_gray_out)
{
#ifdef __ARM_NEON
  // NEON optimized version
  // Using integer arithmetic: 0.299*R + 0.587*G + 0.114*B
  // Scaled to fixed point: (77*R + 150*G + 29*B) >> 8
  // This approximates the float math with integers
  
  const uint8x8_t coeff_b = vdup_n_u8(29);   // 0.114 * 256 ≈ 29
  const uint8x8_t coeff_g = vdup_n_u8(150);  // 0.587 * 256 ≈ 150
  const uint8x8_t coeff_r = vdup_n_u8(77);   // 0.299 * 256 ≈ 77
  
  for (int i = 0; i < img.rows; i++) {
    unsigned char* src_row = img.data + STEP0 * i;
    unsigned char* dst_row = img_gray_out.data + IMG_WIDTH * i;
    
    int j = 0;
    // Process 8 pixels at a time
    for (; j <= img.cols - 8; j += 8) {
      // Load 8 BGR pixels (24 bytes total)
      // Deinterleave BGR into separate channels
      uint8x8x3_t bgr = vld3_u8(&src_row[STEP1 * j]);
      
      // bgr.val[0] = B channel (8 pixels)
      // bgr.val[1] = G channel (8 pixels)
      // bgr.val[2] = R channel (8 pixels)
      
      // Multiply each channel by its coefficient
      uint16x8_t b_weighted = vmull_u8(bgr.val[0], coeff_b);
      uint16x8_t g_weighted = vmull_u8(bgr.val[1], coeff_g);
      uint16x8_t r_weighted = vmull_u8(bgr.val[2], coeff_r);
      
      // Add all channels together
      uint16x8_t sum = vaddq_u16(b_weighted, g_weighted);
      sum = vaddq_u16(sum, r_weighted);
      
      // Shift right by 8 to divide by 256 (return to normal range)
      uint8x8_t gray = vshrn_n_u16(sum, 8);
      
      // Store 8 grayscale pixels
      vst1_u8(&dst_row[j], gray);
    }
    
    // Handle remaining pixels with scalar code
    for (; j < img.cols; j++) {
      double color = .114 * src_row[STEP1*j] +
                     .587 * src_row[STEP1*j + 1] +
                     .299 * src_row[STEP1*j + 2];
      dst_row[j] = color;
    }
  }
#else
  // Scalar fallback
  double color;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
    }
  }
#endif
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