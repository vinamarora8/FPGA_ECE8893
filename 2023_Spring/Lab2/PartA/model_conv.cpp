///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    model_conv.cpp
// Description: Create C model convolution for functional correctness.
//
// TODO: Implement the 7x7 convolution any way you like!
///////////////////////////////////////////////////////////////////////////////

#include "conv.h"

#define KERN_HEIGHT 7
#define KERN_WIDTH 7

static float relu(float x)
{
    if (x < 0)
      return 0.0;
    else
      return x;
}

void model_conv (
    fm_t input_feature_map[3][736][1280],
    wt_t layer_weights[64][3][7][7],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][368][640]
)
{
//--------------------------------------------------------------------------
// Your code for TASK A goes here
//
// Implement the 7x7 convolution layer in typical C/C++ manner.
// You are free to use any C/C++ programming constructs which may
// or may not be HLS-friendly.
//
// The sole purpose of this code is to get the functionality right!
//
// Hints:
// - Handle stride and border pixels appropriately.
// - Do not forget to add bias and apply ReLU!
//--------------------------------------------------------------------------

    for (int of = 0; of < OUT_FM_DEPTH; of++)
      for (int oh = 0; oh < OUT_FM_HEIGHT; oh++)
        for (int ow = 0; ow < OUT_FM_WIDTH; ow++)
        {

            output_feature_map[of][oh][ow] = 0;

            float sum = 0;

            for (int id = 0; id < IN_FM_DEPTH; id++)
              for (int kh = 0; kh < KERN_HEIGHT; kh++)
                for (int kw = 0; kw < KERN_WIDTH; kw++)
                {

                    int idx_h = (STRIDE * oh) - PADDING + kh;
                    int idx_w = (STRIDE * ow) - PADDING + kw;

                    if (idx_h < 0 || idx_w < 0)
                      continue;

                    if (idx_h >= IN_FM_HEIGHT || idx_w >= IN_FM_WIDTH)
                      continue;

                    sum += input_feature_map[id][idx_h][idx_w] * layer_weights[of][id][kh][kw];

                }


            output_feature_map[of][oh][ow] += relu(sum + layer_bias[of]);
        }

}
