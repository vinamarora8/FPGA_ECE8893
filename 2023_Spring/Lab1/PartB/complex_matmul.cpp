///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    complex_matmul.cpp
// Description: Perform matrix multiplication with complex values
//
// Note:        You are free to modify this code to implement your design.
///////////////////////////////////////////////////////////////////////////////

#include "complex.h"

void complex_matmul(
    complex_t MatA_DRAM[M][N],
    complex_t MatB_DRAM[N][K],
    complex_t MatC_DRAM[M][K]
)
{
#pragma HLS interface m_axi depth=1 port=MatA_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatB_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=1 port=MatC_DRAM offset=slave bundle=mem

#pragma HLS interface s_axilite port=return

    // TODO: Insert your code here
    // You may follow a similar style as in real_matmul.cpp

    complex_t MatA[M][N];
    complex_t MatB[N][K];
    complex_t MatC[M][K];

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

    // Initialize product matrix C
    MAT_C_ROWS_INIT:
    for(int i = 0; i < M; i++) {
        MAT_C_COLS_INIT:
        for(int j = 0; j < K; j++) {
            MatC[i][j] = {0, 0};
        }
    }

    // Perform matrix multiplication
    OUTER_ROWS:
    for(int i = 0; i < M; i++) {
        OUTER_COLS:
        for(int j = 0; j < K; j++) {
            INNER_ROW_COL:
            for(int p = 0; p < N; p++) {
                MatC[i][j].real +=   MatA[i][p].real * MatB[p][j].real;
                MatC[i][j].real += - MatA[i][p].imag * MatB[p][j].imag;

                MatC[i][j].imag +=   MatA[i][p].real * MatB[p][j].imag;
                MatC[i][j].imag +=   MatA[i][p].imag * MatB[p][j].real;
            }
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
