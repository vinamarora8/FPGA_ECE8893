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

    #pragma HLS ARRAY_RESHAPE variable=X_buf type=block factor=3 dim=1
    #pragma HLS ARRAY_RESHAPE variable=W_buf type=block factor=3 dim=2

    #pragma HLS ARRAY_PARTITION variable=W_buf type=complete dim=4

    //#pragma HLS ARRAY_PARTITION variable=X_buf type=block factor=4 dim=2
    //#pragma HLS ARRAY_PARTITION variable=Y_buf type=block factor=4 dim=2

    //#pragma HLS ARRAY_PARTITION variable=W_buf type=block factor=7 dim=4

    const int S = STRIDE;

    OUT_FEAT:
    for (int of = 0; of < OUT_BUF_DEPTH; of++)
    {
        OUT_ROW:
        for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)
        {
            OUT_COL:
            for (int ow = 0; ow < OUT_BUF_WIDTH; ow++)
            {
                fm_t sum = 0;
                IN_ROW:
                for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
                {
                    //fm_t sum_local = 0;
                    IN_COL:
                    for (int kw = 0; kw < KERNEL_WIDTH; kw++)
                    {
                        IN_FEAT:
                        for (int id = 0; id < IN_BUF_DEPTH; id++)
                        {
                            #pragma HLS unroll // Covered by array reshape
                            Y_buf[of][oh][ow] += X_buf[id][S*oh + kh][S*ow + kw] * W_buf[of][id][kh][kw];
                        }
                    }
                }
            }
        }
    }
}
