---
title: Introduction to CUDA [2]
subtitle: CUDA memory allocation syntaxes
categories: Programming
tags: CUDA C C++
toc: true
toc_sticky: true
---

# Host-device interaction

이전 포스팅에서 다루었듯이, host와 device는 물리적으로 분리되어 있다. 따라서, host와 device는 독립적인 메모리를 가지고 있으며, memory allocation 역시 host와 device에서 각각 진행해야 한다. 뿐만 아니라, host와 device 사이에서 데이터를 교환하는 것 역시 자명하지 않다. 이번 포스팅에서는 host와 device에서의 memory allocation과 데이터 교환을 위해 제공되는 CUDA C/C++ 함수들을 다룰 예정이다. 매우 간단한 예제를 통해서 중요한 몇몇 함수들을 배워 보자.

# CUDA C/C++ Basic Example (3): Adding with GPU

다음 예제 코드는 매우 간단한 덧셈 연산을 GPU에서 수행하는 코드이다. 중요한 것은, 사용자가 키보드를 통해 입력하는 input은 host의 memory 위로 올라간다는 사실이다. GPU는 이 input을 알지 못하며, 덧셈을 위한 두 input 숫자를 명시적으로 GPU 메모리로 전송시켜 주어야 한다.

GPU 메모리로 이 숫자들을 무작정 전달하는 것이 아니라, 숫자들이 들어갈 메모리 공간을 미리 allocate해야 한다. 이 점을 고려해서 코드를 살펴 보자. 해당 코드는 "예제로 배우는 CUDA 프로그래밍" 책의 코드를 거의 참고해 작성되었다. [GitHub repo로 이동](https://github.com/kadryjh1724/cudaExamples/blob/main/part2/code02-GPUadd.cu)

```c
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int a, b, c;
    int *d_c;
    printf("Input two integers: ");
    scanf("%d %d", &a, &b);
    // Assign device memory
    cudaMalloc((void**)&d_c, sizeof(int));
    // Kernel call
    add<<<1,1>>>(a, b, d_c);
    // Copy the result
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d = %d\n", a, b, c);
    // Free device memory
    cudaFree(d_c);
    return 0;
}
```

이 코드 안에는 