# CEPARCO_CUDA
### CEPARCO S11 CUDA Programming Group Project that computes the absolute sum of a large array using multiple CUDA implementations.

# Group Members:
### Gian Martin Carlos
### Kenzo Mikael De Vera
### Hans Christian Mascardo
### Lucas Antonio Tujan

# Project Specification:
<img width="1296" height="135" alt="image" src="https://github.com/user-attachments/assets/1085c4c6-294e-44aa-8ad7-d19f48795fa8" />

# AI Usage Declaration:

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
| 2^28 Elements. CUDA block size = 1024       | Kernel time | Speedup vs baseline C program |
| -------:                                    | :------:    | :-------                      |
| x86-64                                      |             |                               |
| x86-64 SIMD XMM                             |             |                               |
| x86-64 SIMD YMM                             |             |                               |
| CUDA Unified                                |             |                               |
| CUDA Prefetch                               |             |                               |
| CUDA Prefetch + Page Creation               |             |                               |
| CUDA Prefetch + Page Creation + Mem Advise  |             |                               |
| CUDA Classic MEMCPY                         |             |                               |
| CUDA Data Init. in a CUDA Kernel            |             |                               |

