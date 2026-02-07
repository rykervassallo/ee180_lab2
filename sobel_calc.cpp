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
void grayScale(Mat& img, Mat& img_gray_out)
{
#ifdef __ARM_NEON
  const uint8x16_t coeff_b = vdupq_n_u8(29);   // 0.114 * 256 ≈ 29
  const uint8x16_t coeff_g = vdupq_n_u8(150);  // 0.587 * 256 ≈ 150
  const uint8x16_t coeff_r = vdupq_n_u8(77);   // 0.299 * 256 ≈ 77
  
  for (int i = 0; i < img.rows; i++) {
    unsigned char* src_row = img.data + STEP0 * i;
    unsigned char* dst_row = img_gray_out.data + IMG_WIDTH * i;
    
    int j = 0;
    // Process 16 pixels at a time
    for (; j <= img.cols - 16; j += 16) {
      // Load 16 pixels
      uint8x16x3_t bgr = vld3q_u8(&src_row[STEP1 * j]);
      
      uint8x8_t b_low = vget_low_u8(bgr.val[0]);
      uint8x8_t g_low = vget_low_u8(bgr.val[1]);
      uint8x8_t r_low = vget_low_u8(bgr.val[2]);
      
      uint16x8_t b_weighted_low = vmull_u8(b_low, vget_low_u8(coeff_b));
      uint16x8_t g_weighted_low = vmull_u8(g_low, vget_low_u8(coeff_g));
      uint16x8_t r_weighted_low = vmull_u8(r_low, vget_low_u8(coeff_r));
      
      uint16x8_t sum_low = vaddq_u16(b_weighted_low, g_weighted_low);
      sum_low = vaddq_u16(sum_low, r_weighted_low);
      
      uint8x8_t gray_low = vshrn_n_u16(sum_low, 8);
      
      uint8x8_t b_high = vget_high_u8(bgr.val[0]);
      uint8x8_t g_high = vget_high_u8(bgr.val[1]);
      uint8x8_t r_high = vget_high_u8(bgr.val[2]);
      
      uint16x8_t b_weighted_high = vmull_u8(b_high, vget_high_u8(coeff_b));
      uint16x8_t g_weighted_high = vmull_u8(g_high, vget_high_u8(coeff_g));
      uint16x8_t r_weighted_high = vmull_u8(r_high, vget_high_u8(coeff_r));
      
      uint16x8_t sum_high = vaddq_u16(b_weighted_high, g_weighted_high);
      sum_high = vaddq_u16(sum_high, r_weighted_high);
      
      uint8x8_t gray_high = vshrn_n_u16(sum_high, 8);
      
      uint8x16_t gray = vcombine_u8(gray_low, gray_high);
      
      vst1q_u8(&dst_row[j], gray);
    }
    
    for (; j < img.cols; j++) {
      double color = .114 * src_row[STEP1*j] +
                     .587 * src_row[STEP1*j + 1] +
                     .299 * src_row[STEP1*j + 2];
      dst_row[j] = color;
    }
  }
#else

  // Otherwise just do scalar
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
#ifdef __ARM_NEON
  for (int i = 1; i < img_gray.rows - 1; i++) {
    unsigned char* row_prev = img_gray.data + IMG_WIDTH * (i - 1);
    unsigned char* row_curr = img_gray.data + IMG_WIDTH * i;
    unsigned char* row_next = img_gray.data + IMG_WIDTH * (i + 1);
    unsigned char* out_row = img_sobel_out.data + IMG_WIDTH * i;
    
    int j = 1;
    // Process 8 pixels at a time 
    for (; j <= img_gray.cols - 9; j += 8) {
      // Load the 3x10 pixel block 
      uint8x8_t p00 = vld1_u8(&row_prev[j - 1]);
      uint8x8_t p01 = vld1_u8(&row_prev[j]);
      uint8x8_t p02 = vld1_u8(&row_prev[j + 1]);
      
      uint8x8_t p10 = vld1_u8(&row_curr[j - 1]);
      uint8x8_t p12 = vld1_u8(&row_curr[j + 1]);
      
      uint8x8_t p20 = vld1_u8(&row_next[j - 1]);
      uint8x8_t p21 = vld1_u8(&row_next[j]);
      uint8x8_t p22 = vld1_u8(&row_next[j + 1]);
      
      int16x8_t s00 = vreinterpretq_s16_u16(vmovl_u8(p00));
      int16x8_t s01 = vreinterpretq_s16_u16(vmovl_u8(p01));
      int16x8_t s02 = vreinterpretq_s16_u16(vmovl_u8(p02));
      int16x8_t s10 = vreinterpretq_s16_u16(vmovl_u8(p10));
      int16x8_t s12 = vreinterpretq_s16_u16(vmovl_u8(p12));
      int16x8_t s20 = vreinterpretq_s16_u16(vmovl_u8(p20));
      int16x8_t s21 = vreinterpretq_s16_u16(vmovl_u8(p21));
      int16x8_t s22 = vreinterpretq_s16_u16(vmovl_u8(p22));
      
      // Gx = (p02 + 2*p21 + p22) - (p00 + 2*p01 + p20)
      int16x8_t gx = vsubq_s16(
        vaddq_s16(vaddq_s16(s02, s22), vshlq_n_s16(s21, 1)),
        vaddq_s16(vaddq_s16(s00, s20), vshlq_n_s16(s01, 1))
      );
      
      // Gy = (p20 + 2*p12 + p22) - (p00 + 2*p10 + p02)
      int16x8_t gy = vsubq_s16(
        vaddq_s16(vaddq_s16(s20, s22), vshlq_n_s16(s12, 1)),
        vaddq_s16(vaddq_s16(s00, s02), vshlq_n_s16(s10, 1))
      );
      
      gx = vabsq_s16(gx);
      gy = vabsq_s16(gy);
      
      uint8x8_t result = vqmovun_s16(vaddq_s16(gx, gy));
      
      vst1_u8(&out_row[j], result);
    }
    
    // Scalar for the rest
    for (; j < img_gray.cols - 1; j++) {
      int p00 = row_prev[j-1], p01 = row_prev[j], p02 = row_prev[j+1];
      int p10 = row_curr[j-1], p12 = row_curr[j+1];
      int p20 = row_next[j-1], p21 = row_next[j], p22 = row_next[j+1];
      
      int sobel_x = abs(p00 - p20 + 2*p01 - 2*p21 + p02 - p22);
      int sobel_y = abs(p00 - p02 + 2*p10 - 2*p12 + p20 - p22);
      
      int sobel = sobel_x + sobel_y;
      out_row[j] = (sobel > 255) ? 255 : sobel;
    }
  }
#else

  for (int i = 1; i < img_gray.rows - 1; i++) {
    unsigned char* row_prev = img_gray.data + IMG_WIDTH * (i - 1);
    unsigned char* row_curr = img_gray.data + IMG_WIDTH * i;
    unsigned char* row_next = img_gray.data + IMG_WIDTH * (i + 1);
    unsigned char* out_row = img_sobel_out.data + IMG_WIDTH * i;
    
    unsigned char p00 = row_prev[0], p01 = row_prev[1], p02 = row_prev[2];
    unsigned char p10 = row_curr[0], p12 = row_curr[2];
    unsigned char p20 = row_next[0], p21 = row_next[1], p22 = row_next[2];
    
    for (int j = 1; j < img_gray.cols - 1; j++) {
      int sobel_x = abs(p00 - p20 + 2*p01 - 2*p21 + p02 - p22);
      int sobel_y = abs(p00 - p02 + 2*p10 - 2*p12 + p20 - p22);
      
      int sobel = sobel_x + sobel_y;
      out_row[j] = (sobel > 255) ? 255 : sobel;
      
      p00 = p01; p01 = p02; p02 = row_prev[j + 2];
      p10 = p12; p12 = row_curr[j + 2];
      p20 = p21; p21 = p22; p22 = row_next[j + 2];
    }
  }
#endif
}