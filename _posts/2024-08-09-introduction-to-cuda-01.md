---
title: Introduction to CUDA [1]
subtitle: Concept of GPUs and CUDA hello world code
author: Jiho Son
categories: Programming
tags: CUDA C C++
toc: true
toc_sticky: true
---

# Introduction to CUDA C/C++

CUDA C/C++를 공부하고 간단한 계산에 적용해 보는 과정을 포스팅으로 남길 예정이다. CUDA C/C++를 공부하는 방법에는 여러 가지가 있다. 시중에 CUDA C/C++를 다루는 책도 여러 권 나와 있다 (한국어 화자인 나 역시 처음에는 이런 책들을 참고해 공부하고 있다). 하지만 무엇보다 CUDA C/C++를 공부할 때의 golden standard는 NVIDIA가 제공하는 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)를 참고하는 것인 만큼, 조금 CUDA와 친해졌다면 공식 매뉴얼을 참조해 공부하도록 하자.

# GPGPU (General Purpose Graphic Processing Unit)

GPU (Graphic Processing Unit)이라는 이름이 생긴 이유는, computer graphics를 위한 계산에 특화된 장치로서 GPU가 개발되었기 때문이다. 컴퓨터 화면의 각 픽셀의 RGB 값은 서로 독립적으로 계산할 수 있으며, 병렬적으로 일할 수 있는 worker의 수가 늘어나면 화면을 렌더링하는 데 소요되는 시간이 필연적으로 감소한다. 따라서 초창기의 GPU는 굉장히 한정된 종류의 작업만 가능한 device였다. 하지만 시간이 지나며 GPU의 강력한 병렬 연산 능력을 계산화학과 전산유체역학 등등 다른 분야에서도 활용하게 되었다. 이러한 면에서 general-purpose GPU라고 부르게 되는 것이다.  

> CUDA는 NVIDIA에서 개발한 프로그래밍 인터페이스로, GPU를 사용자가 원하는 목적대로 사용할 수 있도록 유연한 기능을 제공한다. 따라서, NVIDIA GPU를 가지고 있어야 CUDA를 배우는 의미가 있으며, CUDA C/C++로 작성된 코드를 컴파일하려면 당연히 `nvcc` 컴파일러와 여러 가지 GPU 드라이버 등등 필요한 소프트웨어가 설치되어 있어야 한다. 이런 기본적인 설정은 이 포스트에서 다루지 않는다.

컴퓨터의 핵심이자 두뇌의 역할을 하는 것은 바로 중앙처리장치 (CPU)이다. 우리는 CPU를 **host**, GPU를 **device**로 호칭할 것이다. GPU는 혼자서는 동작하지 않으며, CPU로부터 정보를 전달받아 연산을 수행하고, 그 결과를 다시 CPU로 돌려보낸다. 즉 host와 device는 서로 데이터를 주고받으며 계산을 진행하고, 이것을 heterogeneous computing scheme이라고 말할 수 있다. CUDA C/C++에서도 host와 device에서 실행되는 명령과, host와 device의 memory allocation을 구분하는 것이 중요한 포인트이다.  

# SIMT (Single Instruction Multiple Thread) 구조

GPU가 어떤 방식으로 여러 개의 일을 동시에 처리하는지 자세히 알아보자. GPU는 SIMT (단일 명령, 여러 개의 스레드) 구조를 가지고 있다.

> 스레드 (thread)란 운영 체제 내에서 연산 장치를 활용하는 **기본 단위**를 일컫는다. 즉 코어를 사용해서 연산을 수행하는 어떠한 주체 하나를 1개의 스레드라고 생각하면 될 것이다.

GPU가 SIMT 구조를 가지고 있다는 것은 대략적으로:
- 하나의 명령으로 여러 개의 스레드가 여러 데이터에 연산을 수행한다. 즉, 데이터마다 다른 연산을 진행할 수는 없다.
- 반면, 각 스레드는 독립된 제어 문맥을 가진다.

# CUDA C/C++ Basic Example (1): CUDA Hello World!

누구나 C를 처음 배울 때 내가 입력하는 코드가 무슨 의미인지 잘 모르는 상태에서 따라 친 후 컴파일과 실행을 해보기 마련이다. 이번에도 비슷하게 해보자. [GitHub repo로 이동](https://github.com/kadryjh1724/cudaExamples/blob/main/part1/code00-hellocuda.cu)

```c
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void helloCUDA(void) {
    printf("Hello world from GPU, thread %d!\n", threadIdx.x);
}

int main(void) {
    printf("Hello world from CPU!\n");
    helloCUDA<<<1, 16>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

- Line 2-3은 CUDA 관련 syntax를 사용하기 위해서 필요한 헤더 파일을 포함시키는 과정이다.
- Line 9-14에는 익숙한 main 함수가 작성되어 있다. Line 10은 기존의 C의 hello world를 출력하는 함수이다. 
- 반면, 그 이후 Line 11에서는 C 문법에서는 등장하지 않았던 syntax를 가지는 **커널 (kernel)**이 호출되는 것을 볼 수 있다. 이 커널은 Line 5-7에 정의되어 있다.
- Line 6에는 `threadIdx.x`라는 처음 보는 변수가 등장한다. 이것의 의미는 나중에 다시 알아보는 것으로 두고, 덮어두고 넘어가자.

Line 11과 같은 함수를 **커널 함수**라고 부른다. 이 함수의 정의 시 `__global__` 키워드가 붙는 것을 확인할 수 있는데, 이것은 이 함수가 host와 device 양쪽에서 모두 호출이 가능하다는 의미를 가진다. 만약 어떤 커널이 `__host__`나 `__device__` 키워드를 가지고 있다면, 이 커널은 각각 host와 device에서만 호출되고 실행될 수 있다는 의미를 가진다. 여기서 `helloCUDA()` 커널은 host에서 호출되어, device에서 실행된다.

커널 함수는 <<< >>>라는, 본 적 없는 syntax를 가지고 있다. 이 부분은 실행 구성 (execution configuration) 문법이라고 부른다. 지금 수준에서 간단하게만 설명하면, 이 꺽쇠 괄호 내부에 커널을 수행할 스레드의 개수를 입력하는 것이라고 보면 된다. 즉 `<<<1, 16>>>`이란 16개의 스레드를 이용하겠다는 의미를 가진다 (앞의 1이 어떤 의미를 가지는지는 나중의 포스트에서 확인할 예정이다). 이제 저 코드를 컴파일하자.

```bash
nvcc code00-hellocuda.cu -o code00.o
```

`nvcc` 컴파일러는 컴파일할 1개 이상의 파일을 전달받는다. 컴파일이 완료된 output을 명시하기 위해 `-o` flag를 이용한다. 컴파일된 결과물을 실행하기 위해서는 다음과 같이 터미널에 입력하자 (GitHub repository에는 executable file은 업로드되어 있지 않으므로 직접 compile해야 한다).

```bash
./code00.o
```

실행한 결과는 다음과 같다.
```
Hello world from CPU!
Hello world from GPU, thread 0!
Hello world from GPU, thread 1!
Hello world from GPU, thread 2!
Hello world from GPU, thread 3!
Hello world from GPU, thread 4!
Hello world from GPU, thread 5!
Hello world from GPU, thread 6!
Hello world from GPU, thread 7!
Hello world from GPU, thread 8!
Hello world from GPU, thread 9!
Hello world from GPU, thread 10!
Hello world from GPU, thread 11!
Hello world from GPU, thread 12!
Hello world from GPU, thread 13!
Hello world from GPU, thread 14!
Hello world from GPU, thread 15!
```

먼저 Line 10의 hello world (CPU)가 출력된다. 그 이후, `helloCUDA<<<1, 16>>>()` 커널이 실행된다. 우리는 16개의 스레드를 사용하기로 했으므로, Hello world from GPU라는 메시지는 총 16번 출력된다. 우리는 이제 가장 간단한 CUDA 커널과 프로그램을 작성했고, GPU에게 명령을 내리는 방법을 배웠다!

# CUDA C/C++ Basic Example (2): Check your GPU!

내가 가지고 있는 GPU가 얼마나 강력한 컴퓨팅 성능을 제공하는지 확인하는 것은 중요하다. 추후 다시 언급하겠지만, GPU의 종류에 따라 내부에 들어 있는 CUDA 코어의 개수, 메모리 (공유 메모리, 지역 메모리, ...)와 캐시의 크기 및 기타 등등 사양이 달라진다. 이런 하드웨어적 제한을 잘 알고 있어야 GPU를 더욱 효율적으로 사용할 수 있다. 물론 각 GPU의 사양은 NVIDIA사가 제공하는 카탈로그에도 잘 나와 있지만, CUDA C/C++의 인터페이스를 통해서도 많은 정보를 확인할 수 있다. 이번 코드에서는 내 GPU의 사양을 확인하는 방법을 다뤄 본다. [GitHub repo로 이동](https://github.com/kadryjh1724/cudaExamples/blob/main/part1/code01-gpucheck.cu)

```c
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(void) {
    cudaDeviceProp prop;

    // Get total device number
    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("===== ===== ===== ===== ===== ===== ===== ===== =====\n");
        printf("----- [Device %02d]       %s\n", i, prop.name);
        printf("Compute capability      : %d.%d\n", prop.major, prop.minor);
        printf("Clock rate              : %d\n", prop.clockRate);
        printf("----- [Device %02d]       Memory information\n", i);
        printf("Total global memory     : %ld MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("Total const. memory     : %ld KB\n", prop.totalConstMem / 1024);
        printf("L2 Cache size           : %ld KB\n", prop.l2CacheSize / 1024);
        printf("----- [Device %02d]       Multiprocessor information\n", i);
        printf("Multiprocessor count    : %d\n", prop.multiProcessorCount);
        printf("Shared memory per block : %ld KB\n", prop.sharedMemPerBlock / 1024);
        printf("Shared memory per mp    : %ld KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("Registers per mp        : %d\n", prop.regsPerBlock);
        printf("Num. of threads in warp : %d\n", prop.warpSize);
        printf("Max threads per block   : %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dim.         : (%ld, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dim.           : (%ld, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("===== ===== ===== ===== ===== ===== ===== ===== =====\n\n");
    }
    return 0;
}
```

CUDA C/C++에서는 `cudaDeviceProp` 구조체를 통해 device의 사양을 가져오는 기능을 제공한다. 위의 코드에 등장하지 않는 특성도 많이 있으며, 이는 [CUDA runtime api documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1983c292e2078dd5a4240f49c41d647f3)에서 확인할 수 있다.

사실 위에서 출력되는 여러 값들의 의미와 중요성을 이해하려면 GPU의 구조에 대한 이해가 필요하다. 지금 당장은 위와 같은 방식으로 GPU 정보를 가져올 수 있다고 기억만 하고 넘어가도록 하자. 위의 코드도 마찬가지로 컴파일한 후 실행하자.

```bash
nvcc code01-gpucheck.cu -o code01.o
./code01.o
```

내가 가지고 있는 GPU의 종류에 따라 출력은 달라진다. 필자의 lab server main node에서 코드를 실행한 결과는 다음과 같다.

```
===== ===== ===== ===== ===== ===== ===== ===== =====
----- [Device 00]       NVIDIA RTX A2000 12GB
Compute capability      : 8.6
Clock rate              : 1200000
----- [Device 00]       Memory information
Total global memory     : 12032 MB
Total const. memory     : 64 KB
L2 Cache size           : 3072 KB
----- [Device 00]       Multiprocessor information
Multiprocessor count    : 26
Shared memory per block : 48 KB
Shared memory per mp    : 100 KB
Registers per mp        : 65536
Num. of threads in warp : 32
Max threads per block   : 1024
Max thread dim.         : (1024, 1024, 64)
Max grid dim.           : (2147483647, 65535, 65535)
===== ===== ===== ===== ===== ===== ===== ===== =====
```

만약 GPU가 여러 개 있는 서버에 접속 중이라면, 여러 device의 정보가 순서대로 출력될 것이다.

다음 포스팅에서는 host와 device에서의 memory allocation과 data transfer 문법에 대해 알아보고, 이를 이용한 매우 기초적인 예제를 다룰 예정이다.