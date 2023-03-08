///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    tiled_conv.cpp
// Description: Implement a synthesizable tiling-based convolution for
//              ResNet-50's first 7x7 layer with an HD input image.
//
// TODO: Use your unoptimized code from Part B and apply your favorite pragmas
//       to achieve the target speedup (or higher)!
//
//       Add/remove/modify whatever you like.
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void tile (
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH],
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    int t
);


void tiled_conv (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH]
)
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS.
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights       bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias          bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm_out

    #pragma HLS INTERFACE s_axilite register	port=return

    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf0[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t conv_in_buf1[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];

    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------

    const int num_tiles = N_TILE_ROWS * N_TILE_COLS;


    load_input_tile_block_from_DRAM(conv_in_buf0, input_feature_map, 0);

    TILE:
    for(int t = 0; t < num_tiles-1; t++)
    {
        if ((t & 1) == 0)
        {
            tile(output_feature_map, conv_in_buf0, layer_weights, layer_bias, t);
            load_input_tile_block_from_DRAM(conv_in_buf1, input_feature_map, t+1);
        }
        else
        {
            tile(output_feature_map, conv_in_buf1, layer_weights, layer_bias, t);
            load_input_tile_block_from_DRAM(conv_in_buf0, input_feature_map, t+1);
        }
    }

    tile(output_feature_map, conv_in_buf1, layer_weights, layer_bias, num_tiles-1);

}


void tile (
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH],
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    int t
)
{
    /*
     * Process a tile and store the output
     */

    // Parameters and output buffers double-buffered
    wt_t conv_wt_buf0[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf0[OUT_BUF_DEPTH];
    fm_t conv_out_buf0[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    wt_t conv_wt_buf1[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf1[OUT_BUF_DEPTH];
    fm_t conv_out_buf1[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    const int kernel_groups = OUT_FM_DEPTH / OUT_BUF_DEPTH;

    load_layer_params_from_DRAM(conv_wt_buf0, conv_bias_buf0, layer_weights, layer_bias, 0);
    conv_7x7(conv_out_buf0, conv_in_buf, conv_wt_buf0, conv_bias_buf0);
    load_layer_params_from_DRAM(conv_wt_buf1, conv_bias_buf1, layer_weights, layer_bias, 1);

    KERNEL_GRP:
    for (int k = 1; k < kernel_groups-1; k++)
    {
        if ((k & 1) == 1)
        {
            store_output_tile_to_DRAM(output_feature_map, conv_out_buf0, t, k-1);
            conv_7x7(conv_out_buf1, conv_in_buf, conv_wt_buf1, conv_bias_buf1);
            load_layer_params_from_DRAM(conv_wt_buf0, conv_bias_buf0, layer_weights, layer_bias, k+1);
        }
        else
        {
            store_output_tile_to_DRAM(output_feature_map, conv_out_buf1, t, k-1);
            conv_7x7(conv_out_buf0, conv_in_buf, conv_wt_buf0, conv_bias_buf0);
            load_layer_params_from_DRAM(conv_wt_buf1, conv_bias_buf1, layer_weights, layer_bias, k+1);
        }
    }

    store_output_tile_to_DRAM(output_feature_map, conv_out_buf0, t, kernel_groups-2);
    conv_7x7(conv_out_buf1, conv_in_buf, conv_wt_buf1, conv_bias_buf1);
    store_output_tile_to_DRAM(output_feature_map, conv_out_buf1, t, kernel_groups-1);
}
