///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    real_matmul.cpp
// Description: Perform matrix multiplication with real values
//
// Note:        You are free to modify this code to optimize your design.
///////////////////////////////////////////////////////////////////////////////

#include "real.h"

void real_matmul(
    real_t MatA_DRAM[M][N],
    real_t MatB_DRAM[N][K],
    real_t MatC_DRAM[M][K])
{
#pragma HLS interface m_axi depth=1 port=MatA_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatB_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatC_DRAM offset=slave bundle=mem

#pragma HLS interface s_axilite port=return

    real_t MatA[M][N];
    real_t MatB[N][K];
    real_t MatC[M][K];

#pragma HLS ARRAY_RESHAPE variable=MatA complete dim=2
#pragma HLS ARRAY_RESHAPE variable=MatB complete dim=1


    // Read in the data (Matrix A) from DRAM to BRAM
    MAT_A_ROWS:
    for(int i = 0; i < M; i++) {
        MAT_A_COLS:
        for(int j = 0; j < N; j++) {
            MatA[i][j] = MatA_DRAM[i][j];
        }
    }

    // Read in the data (Matrix B) from DRAM to BRAM
    MAT_B_ROWS:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS:
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j];
        }
    }

    // Perform matrix multiplication
    OUTER_ROWS:
    for(int i = 0; i < M; i++) {
        OUTER_COLS:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL:

            int cijLocal = 0;
            for(int p = 0; p < N; p++) {
                cijLocal += MatA[i][p] * MatB[p][j];
            }
            MatC[i][j] = cijLocal;

        }
    }

    // Write back the data from BRAM to DRAM
    MAT_C_ROWS:
    for(int i = 0; i < M; i++) {
        MAT_C_COLS:
        for(int j = 0; j < K; j++) {
            MatC_DRAM[i][j] = MatC[i][j];
        }
    }

}
