# CEPARCO_CUDA
CEPARCO S11 CUDA Programming Group Project that computes the absolute sum of a large array using multiple CUDA implementations.

### Group Members:
Gian Martin Carlos
Kenzo Mikael De Vera
Hans Christian Mascardo
Lucas Antonio Tujan

### Project Specification:
<img width="1296" height="135" alt="image" src="https://github.com/user-attachments/assets/1085c4c6-294e-44aa-8ad7-d19f48795fa8" />

### AI Usage Declaration:

# Program Outputs:
### (1) C Program
<img width="751" height="128" alt="image" src="https://github.com/user-attachments/assets/edd66e3a-a771-490c-82d6-fe77eef0044e" />

### (2) Grid-Stride Loop; no prefetch, no page creation, no mem advise
<img width="872" height="607" alt="image" src="https://github.com/user-attachments/assets/1bcefebc-1d77-468c-b35a-4d78cf02aeb3" />

### (3) Grid-Stride Loop; with prefetch, no page creation, no mem advise
<img width="878" height="635" alt="image" src="https://github.com/user-attachments/assets/d1f85660-797a-4bcc-b753-f79c4bc3b062" />

### (4) Grid-Stride Loop; with prefetch; with page creation; no mem advise
<img width="876" height="633" alt="image" src="https://github.com/user-attachments/assets/1ee2d865-41aa-4c1a-b0ea-77018a31a0ac" />

### (5) Grid-Stride Loop; with prefetch; with page creation; with mem advise
<img width="881" height="649" alt="image" src="https://github.com/user-attachments/assets/53bc559c-aac4-4228-ba61-294d213791d9" />

### (6) Classic MemCopy (no Unified Memory)
<img width="875" height="509" alt="image" src="https://github.com/user-attachments/assets/f5164fe1-8710-4e12-83a9-361eeed631ab" />

### (7) Grid-Stride Loop with Prefetch and GPU Data Initialization
<img width="885" height="649" alt="image" src="https://github.com/user-attachments/assets/b2340c2e-4a53-489f-9b12-edfe585436dc" />

# nSight Outputs:

# Execution Time Table: 
Baseline C Execution Time: 2.422s
| 2^28 Elements. CUDA block size = 1024       | Kernel time | Speedup vs baseline C program |
| -------:                                    | :------:    | :-------                      |
| x86-64                                      | N/A         | N/A                           |
| x86-64 SIMD XMM                             | 502.25ms    | 4.82x                         |
| x86-64 SIMD YMM                             | 499.5ms     | 4.85x                         |
| CUDA Unified                                |             |                               |
| CUDA Prefetch                               |             |                               |
| CUDA Prefetch + Page Creation               |             |                               |
| CUDA Prefetch + Page Creation + Mem Advise  |             |                               |
| CUDA Classic MEMCPY                         |             |                               |
| CUDA Data Init. in a CUDA Kernel            |             |                               |

# Analysis:
a.)

b.)

c.)

d.)

# Problems Encountered:
The absolute sum function to be implemented in CUDA was not possible by using the same code as C, and would write nothing to the asum variable. This occurs due to parallelism, where multiple threads accessing the same variable causes errors in the resulting output. It is unlike the CUDA square program which updates individual elements on an array, hence no "race condition" is ever met across threads. A solution that was implemented was to make use of the atomicAdd() function. This function simply acts as the "var += val" equivalent in C, or in other words, a summation function. 

One of the major changes needed to be done with the program is when cudaDeviceSynchronize() is called. T he absolute sum is the result of a summation which can be done through the use of the atomicAdd() function. Due to the resulting value stored in a scalar value rather than in an array (as compared to the CUDA square program), the program initially showed incorrect outputs when printing the absolute sum, which was observed to be 10 times larger when variable loope is set to 10. This implies that the program is unable to reset the value of asum to 0.0 after each loop, and thus the solution found was to move cudaDeviceSynchronize() inside the loop. This made the program function correctly and return the expected output, albeit with small differences in absolute sum in values less than 0.01, which happens due to how atomicAdd() does not behave the exact same with computing the sum in C.

Small differences in the output from the error checking was performed regardless, as each CUDA program would output a different result from each other, but all in a small difference margin. 

Lastly, the use of the atomicAdd() functions were not supported unless the compiler was told to use CUDA compute compatibility 6.0, which was done by appending "-arch=sm_60" in the cell for compiling the CUDA programs.
# SIMD vs SIMT:
