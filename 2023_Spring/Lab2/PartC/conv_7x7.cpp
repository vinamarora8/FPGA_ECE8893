///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Implement an optimized 7x7 convolution for a single tile block
//
// TODO: Use your unoptimized code from Part B and apply your favorite pragmas
//       to achieve the target latency (or lower)!
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void conv_7x7 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH]
)
{
//---------------------------------------------------------------------------
// Part C: Optimize your Part B code to achieve an overall latency of
//         less than 750ms without exceeding 100% resource utiliization.
//
// TODO: Your code for Part C goes here.
//---------------------------------------------------------------------------

    // Parallelism across output depth (complete)
    #pragma HLS ARRAY_RESHAPE variable=W_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=B_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=Y_buf type=complete dim=1

    // Parallelism across output width (complete)
    // Increasing this further causes dependencies inside OUT_ROW
    // Leading to II violations and increased latency
    #pragma HLS ARRAY_PARTITION variable=X_buf type=block  factor=23 dim=3
    #pragma HLS ARRAY_RESHAPE   variable=X_buf type=cyclic factor=2 dim=3 // IN: 3x52x(23x2)
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=20 dim=3

    #pragma HLS ARRAY_PARTITION variable=W_buf type=cyclic factor=2 dim=4

    // Parallelism across output height
    //#pragma HLS ARRAY_RESHAPE   variable=X_buf type=cyclic factor=2 dim=2
    //#pragma HLS ARRAY_PARTITION variable=X_buf type=cyclic factor=2 dim=2
    //#pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=2 dim=2


    const int S = STRIDE;

    // Bias copy
    BIAS_ROW:  for (int h = 0; h < OUT_BUF_HEIGHT; h++) {
    BIAS_COL:  for (int w = 0; w < OUT_BUF_WIDTH ; w++) {
    BIAS_FEAT: for (int f = 0; f < OUT_BUF_DEPTH ; f++) {
    #pragma HLS unroll
          Y_buf[f][h][w] = (fm_t) B_buf[f];
    }}}


    IN_ROW:   for (int kh = 0; kh < KERNEL_HEIGHT ; kh++)  { // it: 7
    IN_COL:   for (int kw = 0; kw < KERNEL_WIDTH  ; kw+=2)  { // it: 7
    IN_FEAT:  for (int id = 0; id < IN_BUF_DEPTH  ; id++)  { // it: 3
    OUT_ROW:  for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)  { // it: 23
    #pragma HLS unroll factor=1
    OUT_COL:  for (int ow = 0; ow < OUT_BUF_WIDTH ; ow++) { // it: 20
    #pragma HLS unroll
    OUT_FEAT: for (int of = 0; of < OUT_BUF_DEPTH ; of++) {
    #pragma HLS unroll
    fm_t wt_0 = W_buf[of][id][kh][kw];
    fm_t in_0 = X_buf[id][S*oh + kh][S*ow + kw];

    fm_t wt_1 = kw < 6 ? W_buf[of][id][kh][kw+1] : (fm_t) 0;
    fm_t in_1 = kw < 6 ? X_buf[id][S*oh + kh][S*ow + kw+1] : (fm_t) 0;
    Y_buf[of][oh][ow] += (in_0 * wt_0) + (in_1 * wt_1);
    }
    }
    }
    }
    }
    }
}
