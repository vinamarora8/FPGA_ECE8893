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

    int MatA_real[M][N], MatA_imag[M][N];
    int MatB_real[N][K], MatB_imag[N][K];
    int MatC_real[M][K], MatC_imag[M][K];

#pragma HLS ARRAY_RESHAPE variable=MatA_real complete dim=2
#pragma HLS ARRAY_RESHAPE variable=MatA_imag complete dim=2
#pragma HLS ARRAY_RESHAPE variable=MatB_real complete dim=1
#pragma HLS ARRAY_RESHAPE variable=MatB_imag complete dim=1


    // Read in the data (Matrix A) from DRAM to BRAM
    MAT_A_ROWS:
    for(int i = 0; i < M; i++) {
        MAT_A_COLS:
        for(int j = 0; j < N; j++) {
            MatA_real[i][j] = MatA_DRAM[i][j].real;
            MatA_imag[i][j] = MatA_DRAM[i][j].imag;
        }
    }

    // Read in the data (Matrix B) from DRAM to BRAM
    MAT_B_ROWS:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS:
        for(int j = 0; j < K; j++) {
            MatB_real[i][j] = MatB_DRAM[i][j].real;
            MatB_imag[i][j] = MatB_DRAM[i][j].imag;
        }
    }


    // Perform matrix multiplication
    OUTER_ROWS:
    for(int i = 0; i < M; i++) {
        OUTER_COLS:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL:
            int cij_real = 0;
            int cij_imag = 0;
            for(int p = 0; p < N; p++) {
                cij_real +=   MatA_real[i][p] * MatB_real[p][j];
                cij_real += - MatA_imag[i][p] * MatB_imag[p][j];

                cij_imag +=   MatA_real[i][p] * MatB_imag[p][j];
                cij_imag +=   MatA_imag[i][p] * MatB_real[p][j];
            }
            MatC_real[i][j] = cij_real;
            MatC_imag[i][j] = cij_imag;
        }
    }

    // Write back the data from BRAM to DRAM
    MAT_C_ROWS:
    for(int i = 0; i < M; i++) {
        MAT_C_COLS:
        for(int j = 0; j < K; j++) {
            MatC_DRAM[i][j].real = MatC_real[i][j];
            MatC_DRAM[i][j].imag = MatC_imag[i][j];
        }
    }

}
