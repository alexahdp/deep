#include <stdio.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include "point.cuh"
#include "pointShader.hpp"
#include "lib.hpp"


__global__ void simple_vbo_kernel(PointStruct *point, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    point[i].pos = point[i].pos + point[i].vel;
}


Point::Point(int _count) {
    this->count = _count;
    this->size = sizeof(PointStruct) * _count;
    this->data = (PointStruct*)malloc(this->size);
    
    for (int i = 0; i < this->count; i++) {
        this->data[i].pos = {trand(), trand(), 0};
        this->data[i].vel = {trand() / 500.0f, trand() / 500.0f, 0};
    }
    
    // this->data[0].pos = {0.5, 0, 0};
    // this->data[0].vel = {0, 0.001, 0};
    
    // this->data[1].pos = {-0.75, 0, 0};
    // this->data[1].vel = {0, -0.001, 0};
    
    // this->data[2].pos = {0, 0.5, 0};
    // this->data[2].vel = {0.001, 0, 0};
    
    // this->data[3].pos = {0, -0.5, 0};
    // this->data[3].vel = {-0.001, 0, 0};
    
    this->dptr = NULL;
    
    this->pointShaderProgram = getPointShaderProgram();
}

void Point::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Point::unbindVBO() {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
}

void Point::draw() {
    glUseProgram(this->pointShaderProgram);
    glPointSize(10.0);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointStruct), (GLvoid *)0);
    glDrawArrays(GL_POINTS, 0, this->count);
}

void Point::tick() {
    int THREADS_PER_BLOCK = 1024;
    
    int threads = this->count % THREADS_PER_BLOCK;
    int blocks = (this->count + threads) / threads;
    
    simple_vbo_kernel<<<blocks, threads>>>(this->dptr, this->count);
}
