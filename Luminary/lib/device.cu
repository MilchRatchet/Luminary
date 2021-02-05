#include <stdio.h>
#include "device.h"
#include <cuda_runtime_api.h>

extern "C" void display_gpu_information() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compatibility Version: %d.%d\n",
            prop.major, prop.minor);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Global Memory Available: %zu\n",
            prop.totalGlobalMem / ((1024ul)^3));
        printf("  Single to Double Precision Performance Ratio: %d\n",
            prop.singleToDoublePrecisionPerfRatio);
        printf("  Warp Size: %d\n",
            prop.warpSize);
        printf("  Multiprocessor Count: %d\n",
            prop.multiProcessorCount);
        printf("  Registers per Multiprocessor: %d\n",
            prop.regsPerMultiprocessor);
        printf("  Registers per Block: %d\n",
            prop.regsPerBlock);
        printf("  Maximum number of Threads per Block: %d\n",
            prop.maxThreadsPerBlock);
        printf("  Maximum resident Threads per Multiprocessor: %d\n\n",
            prop.maxThreadsPerMultiProcessor);
    }
}
