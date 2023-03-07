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

    #pragma HLS inline off

    // Parallelism across output depth (complete)
    #pragma HLS ARRAY_RESHAPE variable=W_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=B_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=Y_buf type=complete dim=1

    // Parallelism across output width (complete)
    #pragma HLS ARRAY_PARTITION variable=X_buf type=block  factor=26 dim=2
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=23 dim=2

    // Parallelism across kernel width
    //#pragma HLS ARRAY_PARTITION variable=W_buf type=cyclic factor=2 dim=4
    //#pragma HLS ARRAY_RESHAPE   variable=X_buf type=cyclic factor=2 dim=3
    //#pragma HLS ARRAY_PARTITION variable=W_buf type=cyclic factor=2 dim=3
    //#pragma HLS ARRAY_RESHAPE   variable=X_buf type=cyclic factor=2 dim=2

    // Parallelism across output height
    #pragma HLS ARRAY_RESHAPE   variable=X_buf type=cyclic factor=4 dim=3
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=2 dim=3


    const int S = STRIDE;

    // Bias copy
    BIAS_COL:  for (int w = 0; w < OUT_BUF_WIDTH ; w++) {
    #pragma HLS unroll factor=2 // 2 port memory
    BIAS_ROW:  for (int h = 0; h < OUT_BUF_HEIGHT; h++) {
    #pragma HLS unroll
    BIAS_FEAT: for (int f = 0; f < OUT_BUF_DEPTH ; f++) {
    #pragma HLS unroll
          Y_buf[f][h][w] = (fm_t) B_buf[f];
    }}}


    OUT_COL:  for (int ow = 0; ow < OUT_BUF_WIDTH ; ow+=2) { // it: 20
    IN_FEAT:  for (int id = 0; id < IN_BUF_DEPTH  ; id++)  { // it: 3
    IN_ROW:   for (int kh = 0; kh < KERNEL_HEIGHT ; kh++)  { // it: 7
    IN_COL:   for (int kw = 0; kw < KERNEL_WIDTH  ; kw++)  { // it: 4
    #pragma HLS loop_flatten
    #pragma HLS unroll factor=1
    #pragma HLS pipeline II=1
    OUT_ROW:  for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)  { // it: 23
    #pragma HLS unroll
    OUT_FEAT: for (int of = 0; of < OUT_BUF_DEPTH ; of++) {
    #pragma HLS unroll

    Y_buf[of][oh][ow]   += W_buf[of][id][kh][kw] * X_buf[id][S*oh + kh][S*ow + kw];
    Y_buf[of][oh][ow+1] += W_buf[of][id][kh][kw] * X_buf[id][S*oh + kh][S*(ow+1) + kw];

    /*
    fm_t wt_0, wt_1, wt_2, wt_3;
    fm_t in_0, in_1, in_2, in_3;
    wt_0 = wt_1 = wt_2 = wt_3 = (fm_t) 0;
    in_0 = in_1 = in_2 = in_3 = (fm_t) 0;

    wt_0 = W_buf[of][id][kh][kw];
    in_0 = X_buf[id][S*oh + kh][S*ow + kw];
    */

    /*
    if (kw < 6) {
        wt_1 = W_buf[of][id][kh][kw+1];
        in_1 = X_buf[id][S*oh + kh][S*ow + kw+1];
    }
    */

    /*
    if (kh < 6) {
        wt_2 = W_buf[of][id][kh+1][kw];
        in_2 = X_buf[id][S*oh + kh+1][S*ow + kw];
    }

    if ((kh < 6) && (kw < 6)) {
        wt_3 = W_buf[of][id][kh+1][kw+1];
        in_3 = X_buf[id][S*oh + kh+1][S*ow + kw+1];
    }
    */

    //Y_buf[of][oh][ow] += (in_0 * wt_0) + (in_1 * wt_1) + (in_2 * wt_2) + (in_3 * wt_3);

    }}}}}}
}
