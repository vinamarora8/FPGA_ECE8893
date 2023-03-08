///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Implement a functionally-correct synthesizable 7x7 convolution
//              for a single tile block without any optimizations
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void conv_7x7 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH]
)
{
    /*
    // Part B: Implement a trivial functionally-correct single-tile convolution.
    //
    //         This should have an overall latency in the order of 22-23 seconds.
    //
    //         If it's worse than trivial, it may be worth fixing this first.
    //         Otherwise, achieving the target latency with a worse-than-trivial
    //         baseline may be difficult!
    //
    // TODO: Your code for Part B goes here.
    */

    const int S = STRIDE;

 OUT_FEAT:
    for (int of = 0; of < OUT_BUF_DEPTH; of++)
    OUT_ROW:
      for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++)
      OUT_COL:
        for (int ow = 0; ow < OUT_BUF_WIDTH; ow++)
        IN_FEAT:
          for (int id = 0; id < IN_BUF_DEPTH; id++)
          IN_ROW:
            for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
            IN_COL:
              for (int kw = 0; kw < KERNEL_WIDTH; kw++)
              {
                  if (id == 0 && kh == 0 && kw == 0)
                    Y_buf[of][oh][ow] = B_buf[of];

                  int i = S*oh + kh;
                  int j = S*ow + kw;

                  Y_buf[of][oh][ow] += X_buf[id][i][j] * W_buf[of][id][kh][kw];
              }


}
