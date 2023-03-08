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
    #pragma HLS inline off

    // Parallelism across output depth (complete)
    #pragma HLS ARRAY_RESHAPE variable=W_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=B_buf type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=Y_buf type=complete dim=1

    // Parallelism across output width (complete)
    #pragma HLS ARRAY_PARTITION variable=X_buf type=block  factor=26 dim=2
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=23 dim=2

    // Parallelism across output height
    // X_buf doesn't need it since it already has 2 ports, both can be used for read
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=cyclic factor=2 dim=3

    const int S = STRIDE;

    OUT_COL:  for (int ow = 0; ow < OUT_BUF_WIDTH ; ow+=2) { // it: 20
    IN_FEAT:  for (int id = 0; id < IN_BUF_DEPTH  ; id++)  { // it: 3
    IN_ROW:   for (int kh = 0; kh < KERNEL_HEIGHT ; kh++)  { // it: 7
    IN_COL:   for (int kw = 0; kw < KERNEL_WIDTH  ; kw++)  { // it: 4
    #pragma HLS pipeline II=1
    OUT_ROW:  for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)  { // it: 23
    #pragma HLS unroll
    OUT_FEAT: for (int of = 0; of < OUT_BUF_DEPTH ; of++) {
    #pragma HLS unroll

        if (id == 0 && kh == 0 & kw == 0)
        {
          Y_buf[of][oh][ow]   = B_buf[of];
          Y_buf[of][oh][ow+1] = B_buf[of];
        }

        Y_buf[of][oh][ow]   += W_buf[of][id][kh][kw] * X_buf[id][S*oh + kh][S*ow + kw];
        Y_buf[of][oh][ow+1] += W_buf[of][id][kh][kw] * X_buf[id][S*oh + kh][S*(ow+1) + kw];

    }}}}}}
}
