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
    int_t MatC_real[M][K];
    int_t MatC_imag[M][K];

#pragma HLS ARRAY_RESHAPE variable=MatA complete dim=2
#pragma HLS ARRAY_RESHAPE variable=MatB complete dim=1

    for (int step = 0; step < 4; step++)
    {
        // State machine:
        // Step 0: C.real  = A.real * B.real
        // Step 1: C.imag  = A.imag * B.real
        // Step 2: C.real -= A.imag * B.imag
        // Step 3: C.imag += A.real * B.imag, writeback C

        switch (step)
        {
        case 0: // Copy A.real, B.real
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
          break;

        case 1: // Copy A.imag
          MAT_A_ROWS_I1:
          for(int i = 0; i < M; i++) {
              MAT_A_COLS_I1:
              for(int j = 0; j < N; j++) {
                  MatA[i][j] = MatA_DRAM[i][j].imag;
              }
          }
          break;

        case 2: // Copy B.imag
          MAT_B_ROWS_I1:
          for(int i = 0; i < N; i++) {
              MAT_B_COLS_R2:
              for(int j = 0; j < K; j++) {
                  MatB[i][j] = MatB_DRAM[i][j].imag;
              }
          }
          break;

        case 3: // Copy A.real
          MAT_A_ROWS_R2:
          for(int i = 0; i < M; i++) {
              MAT_A_COLS_R2:
              for(int j = 0; j < N; j++) {
                  MatA[i][j] = MatA_DRAM[i][j].real;
              }
          }
          break;

        }

        // Inner products
        OUTER_ROWS:
        for(int i = 0; i < M; i++) {
            OUTER_COLS:
            for(int j = 0; j < K; j++) {
                #pragma HLS PIPELINE II=1
                INNER_ROW_COL:
                int_t cijLocal = 0;
                for(int p = 0; p < N; p++) {
                    cijLocal += MatA[i][p] * MatB[p][j];
                }

                switch (step)
                {
                case 0: MatC_real[i][j]  = cijLocal; break;
                case 1: MatC_imag[i][j]  = cijLocal; break;
                case 2: MatC_real[i][j] -= cijLocal; break;
                case 3: MatC_imag[i][j] += cijLocal; break;
                }
            }
        }

        if (step == 3)
        {
            // Write back the data from BRAM to DRAM
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < K; j++) {
                    MatC_DRAM[i][j].real = MatC_real[i][j];
                    MatC_DRAM[i][j].imag = MatC_imag[i][j];
                }
            }

        }
    }
}
