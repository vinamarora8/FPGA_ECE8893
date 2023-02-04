#!/usr/bin/python3

import time
import os

M_base = 100
N_base = 150
K_base = 200

os.system("mkdir -p reports/")
f_log = open("reports/dse.log", 'w')

def log(s: str) -> None:

    f_log.write(s)
    f_log.write('\n')
    f_log.flush()
    print(s)

def write_macro(factor: int) -> None:
    ''' Write macros.h file '''

    M = factor * M_base
    N = factor * N_base
    K = factor * K_base

    log(f"M: {M}, N: {N}, K: {K}")

    fstring = f"#pragma once\n\n"
    fstring += f"#define M {M}\n"
    fstring += f"#define N {N}\n"
    fstring += f"#define K {K}\n"

    with open("macros.h", 'w') as f:
        f.write(fstring)


f_runtime = open("reports/runtime.log", 'w')
csynth_path = "complex_proj/solution1/syn/report/csynth.rpt"

fact_list = [1, 2, 4, 8]

for fact in fact_list:

    log(f"Fact: {fact}")
    write_macro(fact)
    log("Running make synth")
    start = time.time()
    os.system("make synth")
    runtime = time.time() - start
    log("Done")
    log(f"Runtime: {runtime}s")
    log("-------------------\n")

    os.system(f"cp {csynth_path} reports/fact{fact}.rpt")
    f_runtime.write(f"Factor: {fact}, runtime: {runtime}s\n")

f_log.close()
f_runtime.close()
