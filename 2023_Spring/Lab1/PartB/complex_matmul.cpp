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

    // Internal cache BRAMs, not complete matrices
    int_t MatA[M][N];
    int_t MatB[N][K];
    int_t MatC[M][K];

#pragma HLS ARRAY_RESHAPE variable=MatA complete dim=2
#pragma HLS ARRAY_RESHAPE variable=MatB complete dim=1

    /** C_real **/
    /* A_real * B_real */

    MAT_A_ROWS_R1:
    for(int i = 0; i < M; i++) {
        MAT_A_COLS_R1:
        for(int j = 0; j < N; j++) {
            MatA[i][j] = MatA_DRAM[i][j].real;
        }
    }

    MAT_B_ROWS_R1:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS_R1:
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j].real;
        }
    }

    OUTER_ROWS_RR:
    for(int i = 0; i < M; i++) {
        OUTER_COLS_RR:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL_RR:
            int_t cijLocal = 0;
            for(int p = 0; p < N; p++) {
                cijLocal += MatA[i][p] * MatB[p][j];
            }
            MatC[i][j] = cijLocal;
        }
    }


    /* A_imag * B_imag */

    MAT_A_ROWS_I1:
    for(int i = 0; i < M; i++) {
        MAT_A_COLS_I1:
        for(int j = 0; j < N; j++) {
            MatA[i][j] = MatA_DRAM[i][j].imag;
        }
    }

    MAT_B_ROWS_I1:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS_I1:
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j].imag;
        }
    }

    OUTER_ROWS_II:
    for(int i = 0; i < M; i++) {
        OUTER_COLS_II:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL_II:
            int_t cijLocal = 0;
            for(int p = 0; p < N; p++) {
                cijLocal += MatA[i][p] * MatB[p][j];
            }
            MatC[i][j] -= cijLocal;
        }
    }

    // Write back the data from BRAM to DRAM
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            MatC_DRAM[i][j].real = MatC[i][j];
        }
    }


    /** C_imag **/
    /* A_imag * B_real */

    MAT_B_ROWS_R2:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS_R2:
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j].real;
        }
    }

    OUTER_ROWS_IR:
    for(int i = 0; i < M; i++) {
        OUTER_COLS_IR:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL_IR:
            int_t cijLocal = 0;
            for(int p = 0; p < N; p++) {
                cijLocal += MatA[i][p] * MatB[p][j];
            }
            MatC[i][j] = cijLocal;
        }
    }

    /* A_real * B_imag */

    MAT_A_ROWS_R2:
    for(int i = 0; i < M; i++) {
        MAT_A_COLS_R2:
        for(int j = 0; j < N; j++) {
            MatA[i][j] = MatA_DRAM[i][j].real;
        }
    }

    MAT_B_ROWS_I2:
    for(int i = 0; i < N; i++) {
        MAT_B_COLS_I2:
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j].imag;
        }
    }

    OUTER_ROWS_RI:
    for(int i = 0; i < M; i++) {
        OUTER_COLS_RI:
        for(int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            INNER_ROW_COL_RI:
            int_t cijLocal = 0;
            for(int p = 0; p < N; p++) {
                cijLocal += MatA[i][p] * MatB[p][j];
            }
            MatC[i][j] += cijLocal;
        }
    }

    // Write back the data from BRAM to DRAM
    MAT_C_ROWS_I:
    for(int i = 0; i < M; i++) {
        MAT_C_COLS_I:
        for(int j = 0; j < K; j++) {
            MatC_DRAM[i][j].imag = MatC[i][j];
        }
    }

}
