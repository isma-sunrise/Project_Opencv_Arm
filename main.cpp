// Refactored for build in Ubuntu

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>

// Precise Time Measurement
#if defined(_WIN32) || defined(__CYGWIN__)
// Windows (x86 or x64)
#include <windows.h>
#elif defined(__linux__)
// Linux
#include <time.h>
#else
#error Unknown environment!
#endif

// Intel x86 SIMD Intrinsics
#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

using namespace cv;
using namespace std;

unsigned long calculateElapsedTime(unsigned long start, unsigned long stop);
unsigned long getCurrentTimeInMicroseconds();
void SobelNonSimd(Rect cropSize, Mat inputImage);
void SobelSimd(Rect cropSize, Mat inputImage);
void SobelOpenCV(Rect cropSize, Mat inputImage);

// precise time measurement
unsigned long t1, t2, elapsedTime;

// OpenCV image datatypes
Mat originalImage;
Mat resultImageNonSimd;
Mat resultImageSimd;
Mat resultImageOpenCV;

uchar *outputPointer;
uchar *inputPointer;

int width;
int height;

// crop image and remove added borders
const int border = 8;

int main(int argc, char **argv)
{
    if (border < 1)
    {
        cout << "border must be greater than or equal to 1" << endl;
        cout << "Press any key to exit!" << endl;
        getchar();
        return 0;
    }

    // load input image
    originalImage = imread("monarch.jpg", IMREAD_GRAYSCALE);

    int imageWidth = originalImage.cols;
    int imageHeight = originalImage.rows;

    Rect cropped(border, border, imageWidth, imageHeight);

    // show input image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", originalImage);
    // add border to image using replication method
    copyMakeBorder(originalImage, originalImage, border, border, border, border, BORDER_REPLICATE);

    /******************* Non-SIMD ********************/

    //SobelNonSimd(cropped, originalImage);

    //namedWindow("Sobel-Non-SIMD", WINDOW_AUTOSIZE);
    //imshow("Sobel-Non-SIMD", resultImageNonSimd);

    // /********************** SIMD *********************/

     //SobelSimd(cropped, originalImage);

     //namedWindow("Sobel-SIMD", WINDOW_AUTOSIZE);
     //imshow("Sobel-SIMD", resultImageSimd);

    // /************* OpenCV Built-in Sobel *************/

     SobelOpenCV(cropped, originalImage);

     namedWindow("Sobel-OpenCV", WINDOW_AUTOSIZE);
     imshow("Sobel-OpenCV", resultImageOpenCV);

    waitKey(); // Wait for a keystroke in the window
    getchar();
    return 0;
}

unsigned long calculateElapsedTime(unsigned long start, unsigned long stop)
{
    return stop - start;
}

unsigned long getCurrentTimeInMicroseconds()
{
#if defined(_WIN32) || defined(__CYGWIN__)
    // Windows (x86 or x64)
    LARGE_INTEGER freq;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (unsigned long)((1.0e6 * t.QuadPart) / freq.QuadPart);
#elif defined(__linux__)
    // Linux
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long)(1.0e6 * t.tv_sec + 1.0e-3 * t.tv_nsec);
#else
    return 0;
#error Unknown environment!
#endif
}

void SobelNonSimd(Rect cropSize, Mat inputImage)
{
    Mat outputImage;

    // allocate new array data for outputImage with the same size as inputImage.
    outputImage.create(inputImage.size(), inputImage.depth());

    uchar m1, m2, m3, m4, m5, m6, m7, m8, m9;
    int Gx, Gy;

    outputPointer = outputImage.ptr<uchar>();
    inputPointer = inputImage.ptr<uchar>();

    width = inputImage.cols;
    height = inputImage.rows;

    // start timer
    t1 = getCurrentTimeInMicroseconds();

    for(int i=0; i<100;i++) {
        for (int i = (border - 1); i < height - (border + 1); i++) // rows
        {
            for (int j = (border - 1); j < width - (border + 1); j++) // cols
            {

                /*
                Sobel operator input matrix
                +~~~~~~~~~~~~~~+
                | m1 | m2 | m3 |
                |~~~~+~~~~+~~~~+
                | m4 | m5 | m6 |
                |~~~~+~~~~+~~~~+
                | m7 | m8 | m9 |
                +~~~~+~~~~+~~~~+
                */

                m1 = *(inputPointer + i * width + j);
                m2 = *(inputPointer + i * width + j + 1);
                m3 = *(inputPointer + i * width + j + 2);

                m4 = *(inputPointer + (i + 1) * width + j);
                m5 = *(inputPointer + (i + 1) * width + j + 1);
                m6 = *(inputPointer + (i + 1) * width + j + 2);

                m7 = *(inputPointer + (i + 2) * width + j);
                m8 = *(inputPointer + (i + 2) * width + j + 1);
                m9 = *(inputPointer + (i + 2) * width + j + 2);

                // Calculating Gx
                Gx = (m3 + 2 * m6 + m9) - (m1 + 2 * m4 + m7);

                // Calculating Gy
                Gy = (m1 + 2 * m2 + m3) - (m7 + 2 * m8 + m9);

                outputPointer[(i + 1) * width + j + 1] = saturate_cast<uchar>(abs(Gx) + abs(Gy)); // approximate
            }
        }
    }

    // stop timer
    t2 = getCurrentTimeInMicroseconds();
    // calculate and print elapsed time in microseconds
    elapsedTime = calculateElapsedTime(t1, t2);
    cout << "Execution time for non-SIMD Sobel edge detection:" << endl;
    cout << elapsedTime/100 << " us" << endl;

    outputImage = outputImage(cropSize);

    // Copy outputImage to resultImageNonSimd
    resultImageNonSimd = outputImage.clone();
}

void func(int8_t *buf) {
    int8x16_t vec = vld1q_s8(buf); // load 16x int8_t
    int16x8_t short1 = vmovl_s8(vget_low_s8(vec)); // cast the first 8x int8_t to int16_t
    int16x8_t short2 = vmovl_s8(vget_high_s8(vec)); // cast the last 8x int8_t to int16_t
    short1 = vaddq_s16(short1, short1); // Do operation on int16
    short2 = vaddq_s16(short2, short2);
    vec = vcombine_s8(vmovn_s16(short1), vmovn_s16(short2)); // Cast back to int8_t and combine the two vectors
    vst1q_s8(buf, vec); // Store
}

void print128_num(int16x8_t var)
{
    uint16_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %04X %04X %04X %04X %04X %04X %04X %04X \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7]);
}

void print128_num(int8x16_t var)
{
    uint8_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %04X %04X %04X %04X %04X %04X %04X %04X \n %04X %04X %04X %04X %04X %04X %04X %04X \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], 
           val[14], val[15]);
}

void print128_num(uint8x16_t var)
{
    uint8_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %04X %04X %04X %04X %04X %04X %04X %04X \n %04X %04X %04X %04X %04X %04X %04X %04X \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], 
           val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], 
           val[14], val[15]);
}

void print128_num(int16x4_t var)
{
    uint16_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %04X %04X %04X %04X \n", 
           val[0], val[1], val[2], val[3]);
}

void SobelSimd(Rect cropSize, Mat inputImage)
{
    Mat outputImage;

    // inputImage = imread("boat.png", IMREAD_GRAYSCALE);
    outputImage.create(inputImage.size(), inputImage.depth());

    int8x16_t p1, p2, p3, p4, p5, p6, p7, p8, p9;
    int8x16_t A1, A2, A3, A4, A5, A6, A7, A8, A9;

    uint8x16_t result;
    uint8x16_t gx2, gy2;

    int8x16_t gx, gy, temp, G;
    uint8x16_t vec0;
    outputPointer = outputImage.ptr<uchar>();
    inputPointer = inputImage.ptr<uchar>();

    width = inputImage.cols;
    height = inputImage.rows;

    // start timer
    t1 = getCurrentTimeInMicroseconds();

    for(int i=0; i<100;i++) {
        for (int i = (border - 1); i < height - (border + 1); i += 1)
        {
            for (int j = (border - 1); j < width - (2 * border - 1); j += 8)
            {

                /*
                Sobel operator input matrix
                +~~~~~~~~~~~~~~+
                | p1 | p2 | p3 |
                |~~~~+~~~~+~~~~+
                | p4 | p5 | p6 |
                |~~~~+~~~~+~~~~+
                | p7 | p8 | p9 |
                +~~~~+~~~~+~~~~+
                */

                /*
                __m128i _mm_loadu_si128 (__m128i const* mem_addr)
                Load 128-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
                */

                p1 = vld1q_s8((const int8_t*)(inputPointer + i * width + j));
                p2 = vld1q_s8((const int8_t*)(inputPointer + i * width + j + 1));
                p3 = vld1q_s8((const int8_t*)(inputPointer + i * width + j + 2));

                p4 = vld1q_s8((const int8_t*)(inputPointer + (i + 1) * width + j));
                p5 = vld1q_s8((const int8_t*)(inputPointer + (i + 1) * width + j + 1));
                p6 = vld1q_s8((const int8_t*)(inputPointer + (i + 1) * width + j + 2));

                p7 = vld1q_s8((const int8_t*)(inputPointer + (i + 2) * width + j));
                p8 = vld1q_s8((const int8_t*)(inputPointer + (i + 2) * width + j + 1));
                p9 = vld1q_s8((const int8_t*)(inputPointer + (i + 2) * width + j + 2));

                // vec0 = vcombine_u8(vcreate_u8(0),vcreate_u8(0)) ;

                // p1 = vreinterpretq_s16_u8(vzip1q_u8(A1,vec0));
                // p2 = vreinterpretq_s16_u8(vzip1q_u8(A2,vec0));
                // p3 = vreinterpretq_s16_u8(vzip1q_u8(A3,vec0));
                // p4 = vreinterpretq_s16_u8(vzip1q_u8(A4,vec0));
                // p5 = vreinterpretq_s16_u8(vzip1q_u8(A5,vec0));
                // p6 = vreinterpretq_s16_u8(vzip1q_u8(A6,vec0));
                // p7 = vreinterpretq_s16_u8(vzip1q_u8(A7,vec0));
                // p8 = vreinterpretq_s16_u8(vzip1q_u8(A8,vec0));
                // p9 = vreinterpretq_s16_u8(vzip1q_u8(A9,vec0));


                // if(i == (border - 1) && j == (border - 1)) {
                //     std::cout << "step 0 : ";
                //     print128_num(vec0);
                //     std::cout << "step 1 : ";
                //     print128_num(A1);
                //     std::cout << "step 2 : ";
                //     print128_num(vzip1q_u8(A1,vec0));
                //     std::cout << "step 3 : ";
                //     print128_num(p1);
                //     std::cout << "step 4 : ";
                //     print128_num(vreinterpretq_s16_u8(A1));
                //     // std::cout << "step 3 : ";
                //     // print128_num(p1);
                // }

                /*
                __m128i _mm_srli_epi16 (__m128i a, int imm8)
                Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the Gs in dst.

                __m128i _mm_unpacklo_epi8 (__m128i a, __m128i b)
                Unpack and interleave 8-bit integers from the low half of a and b, and store the Gs in dst.
                */

                // convert image 8-bit unsigned integer data to 16-bit signed integers to use in arithmetic operations

                // p1 = _mm_srli_epi16(_mm_unpacklo_epi8(p1, p1), 8);
                // p2 = _mm_srli_epi16(_mm_unpacklo_epi8(p2, p2), 8);
                // p3 = _mm_srli_epi16(_mm_unpacklo_epi8(p3, p3), 8);
                // p4 = _mm_srli_epi16(_mm_unpacklo_epi8(p4, p4), 8);
                // p5 = _mm_srli_epi16(_mm_unpacklo_epi8(p5, p5), 8);
                // p6 = _mm_srli_epi16(_mm_unpacklo_epi8(p6, p6), 8);
                // p7 = _mm_srli_epi16(_mm_unpacklo_epi8(p7, p7), 8);
                // p8 = _mm_srli_epi16(_mm_unpacklo_epi8(p8, p8), 8);
                // p9 = _mm_srli_epi16(_mm_unpacklo_epi8(p9, p9), 8);

                /*
                __m128i _mm_add_epi16 (__m128i a, __m128i b)
                Add packed 16-bit integers in a and b, and store the Gs in dst.

                __m128i _mm_sub_epi16 (__m128i a, __m128i b)
                Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the Gs in dst.
                */

                // Calculating Gx = (p3 + 2 * p6 + p9) - (p1 + 2 * p4 + p7)
                gx = vaddq_s8(p6, p6);   // 2*p6
                gx = vaddq_s8(gx, p3);   // p3 + 2*p6
                gx = vaddq_s8(gx, p9);   // p3 + 2*p6 + p9
                gx = vsubq_s8(gx, p1);   // p3 + 2*p6 + p9 - p1
                temp = vaddq_s8(p4, p4); // 2*p4
                gx = vsubq_s8(gx, temp); // p3 + 2*p6 + p9 - (p1 + 2*p4)
                gx = vabdq_s8(gx, p7);   // p3 + 2*p6 + p9 - (p1 + 2*p4 + p7)

                // Calculating Gy = (p1 + 2 * p2 + p3) - (p7 + 2 * p8 + p9)
                gy = vaddq_s8(p2, p2);   // 2*p2
                gy = vaddq_s8(gy, p1);   // p1 + 2*p2
                gy = vaddq_s8(gy, p3);   // p1 + 2*p2 + p3
                gy = vsubq_s8(gy, p7);   // p1 + 2*p2 + p3 - p7
                temp = vaddq_s8(p8, p8); // 2*p8
                gy = vsubq_s8(gy, temp); // p1 + 2*p2 + p3 - (p7 + 2*p8)
                gy = vabdq_s8(gy, p9);   // p1 + 2*p2 + p3 - (p7 + 2*p8 + p9)

                /*
                __m128i _mm_abs_epi16 (__m128i a)
                Compute the absolute value of packed 16-bit integers in a, and store the unsigned Gs in dst.
                */

                // gx2 = vabsq_s8(gx); // |Gx|
                // gy2 = vabsq_s8(gy); // |Gy|

                // // G = |Gx| + |Gy|
                G = vqaddq_s8(gx, gy);

                /*
                __m128i _mm_packus_epi16 (__m128i a, __m128i b)
                Convert packed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst.
                */
                // result = vreinterpretq_u8_s8(G);

                /*
                void _mm_storeu_si128 (__m128i* mem_addr, __m128i a)
                Store 128-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
                */
                vst1q_s8((int8_t*)(outputPointer + (i + 1) * width + j + 1), G);
            }
        }
    }

    // stop timer
    t2 = getCurrentTimeInMicroseconds();
    // // calculate and print elapsed time in microseconds
    elapsedTime = calculateElapsedTime(t1, t2);
    cout << "Execution time for SIMD Sobel edge detection:" << endl;
    cout << elapsedTime/100 << " us" << endl;

    // // crop image and remove added borders
    outputImage = outputImage(cropSize);

    // // Copy outputImage to resultImageSimd
    resultImageSimd = outputImage.clone();
}

void SobelOpenCV(Rect cropSize, Mat inputImage)
{
    Mat outputImage;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_8U;

    inputImage = inputImage(cropSize);

    // start timer
    t1 = getCurrentTimeInMicroseconds();
    for(int i=0; i<100;i++) {
        // Gradient X
        Sobel(inputImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(grad_x, abs_grad_x, 1, 0);

        // Gradient Y
        Sobel(inputImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(grad_y, abs_grad_y, 1, 0);

        /// Total Gradient (approximate)
        addWeighted(abs_grad_x, 2.0, abs_grad_y, 2.0, 0, outputImage);
    }
    // stop timer
    t2 = getCurrentTimeInMicroseconds();
    // calculate and print elapsed time in microseconds
    elapsedTime = calculateElapsedTime(t1, t2);
    cout << "Execution time for OpenCV Sobel edge detection:" << endl;
    cout << elapsedTime/100 << " us" << endl;

    // Copy outputImage to resultImageOpenCV
    resultImageOpenCV = outputImage.clone();
}
