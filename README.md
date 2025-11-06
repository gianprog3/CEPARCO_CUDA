# CEPARCO_CUDA
CEPARCO S11 CUDA Programming Group Project that computes the absolute sum of a large array using multiple CUDA implementations.

### Group Members:
Gian Martin Carlos <br>
Kenzo Mikael De Vera <br>
Hans Christian Mascardo <br>
Lucas Antonio Tujan <br>

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
### (2) Grid-Stride Loop; no prefetch, no page creation, no mem advise
<img width="1918" height="1137" alt="nsight_asm1" src="https://github.com/user-attachments/assets/db365f53-fbd6-4770-9c8d-bab9affb0bf7" />

### (3) Grid-Stride Loop; with prefetch, no page creation, no mem advise
<img width="1918" height="1137" alt="nsight_asm2" src="https://github.com/user-attachments/assets/219ee7b3-4e61-4f73-a60e-be083d380e09" />

### (4) Grid-Stride Loop; with prefetch; with page creation; no mem advise
<img width="1918" height="1137" alt="nsight_asm3" src="https://github.com/user-attachments/assets/73a970f6-24ac-4497-9ec0-9ff789235485" />

### (5) Grid-Stride Loop; with prefetch; with page creation; with mem advise
<img width="1918" height="1140" alt="nsight_asm4" src="https://github.com/user-attachments/assets/aaecccd5-e761-4f3f-8c38-15a3c544da01" />

### (6) Classic MemCopy (no Unified Memory)
<img width="1918" height="1137" alt="nsight_asm5" src="https://github.com/user-attachments/assets/cde8b181-a808-42c1-bf89-4cd2cb786911" />

### (7) Grid-Stride Loop with Prefetch and GPU Data Initialization
<img width="1918" height="1140" alt="nsight_asm6" src="https://github.com/user-attachments/assets/7942d7d2-ec98-4b74-9431-7008d1c85630" />


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

# Analysis: <br>
a.) ##What overheads are included in the GPU execution time (up to the point where the data is transferred back to the CPU for error checking)? Is it different for each CUDA variant?
- 
b.) ##How does block size affect execution time (observing various elements and using max blocks)?  Which block size will you recommend?
- The block size determines how many threads can run in parallel on the block. NVIDIA GPUs make use of warps, which executes threads in groups of 32. With this, the block size is recommended to be a multiple of 32 to maximize the warp groups. If the block size is not a multiple of 32, the excess threads would still be on a new warp group, which would waste unallocated space. For example, if we use a block size of 40, there will be 2 groups, 1 with 32 threads and another using only 8 out of the 32 possible threads.<br>
c.) ##Is prefetching always recommended, or should CUDA manage memory?  Give some use cases in which one is better than the other.
- If the programmer is looking to boost performance when using unified memory, then prefetching is always recommended. In comparison to normal unified memory or CUDA's cudaMallocManaged(), prefetching moves the needed pages from the unified memory to the GPU's frames before the kernel runs. This action stops page faulting which would then reduce the time the GPU stalls from retrieving the pages. A use case for the programmer to use normal unified memory is when the program does not need to have optimal performance. An example of this would be during the project's prototype stage, where the programmer could care more about convenience rather than performance.<br>
d.)


# Problems Encountered:
The absolute sum function to be implemented in CUDA was not possible by using the same code as C, and would write nothing to the asum variable. This occurs due to parallelism, where multiple threads accessing the same variable causes errors in the resulting output. It is unlike the CUDA square program which updates individual elements on an array, hence no "race condition" is ever met across threads. A solution that was implemented was to make use of the atomicAdd() function. This function simply acts as the "var += val" equivalent in C, or in other words, a summation function. 

One of the major changes needed to be done with the program is when cudaDeviceSynchronize() is called. T he absolute sum is the result of a summation which can be done through the use of the atomicAdd() function. Due to the resulting value stored in a scalar value rather than in an array (as compared to the CUDA square program), the program initially showed incorrect outputs when printing the absolute sum, which was observed to be 10 times larger when variable loope is set to 10. This implies that the program is unable to reset the value of asum to 0.0 after each loop, and thus the solution found was to move cudaDeviceSynchronize() inside the loop. This made the program function correctly and return the expected output, albeit with small differences in absolute sum in values less than 0.01, which happens due to how atomicAdd() does not behave the exact same with computing the sum in C.

Small differences in the output from the error checking was performed regardless, as each CUDA program would output a different result from each other, but all in a small difference margin. 

Lastly, the use of the atomicAdd() functions were not supported unless the compiler was told to use CUDA compute compatibility 6.0, which was done by appending "-arch=sm_60" in the cell for compiling the CUDA programs.
# SIMD vs SIMT:
While both SIMD and SIMT exhibited a boost in performance compared to our C program, for our use case, SIMD was the better choice. For our kernel, getting the absolute value of every element in the vector would be faster with the use of SIMT, since we could get the value through multiple threads running at the same time. SIMD on the otherhand would only be able to get 16 data values at most and would need additional instructions to cover the entire vector. SIMD's strength comes in the summation aspect since SIMT's parallelism would need synchronization steps to combine each partial sum in the threads, while SIMD can perform the sum directly in registers. Another flaw of SIMT in our kernel would be the passing of data from CPU to GPU and vice versa. Since our kernel wouldn't require complex arithmetic operations, the data transfer times would overpower the time it takes for the SIMT to perform the arithmetic operations. Compared to SIMD, the data is already in the CPU and is readily accessible by the kernel. 
