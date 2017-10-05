#include <stdio.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include "line.cuh"
#include "lineShader.hpp"
#include "lib.hpp"


__global__ void moveLines(LineStruct *line, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    
    line[i].pos.f = line[i].pos.f + make_float3(0.001, 0.001, 0);
    line[i].pos.t = line[i].pos.t + make_float3(0.001, 0.001, 0);
}

Line::Line(int _count) {
    this->count = _count;
    this->size = sizeof(LineStruct) * _count;
    this->data = (LineStruct*)malloc(this->size);
    
    for (int i = 0; i < this->count; i++) {
        this->data[i].pos.f = {trand(), trand(), 0};
        this->data[i].pos.t = {trand(), trand(), 0};
    }
    
    this->dptr = NULL;
    this->lineShaderProgram = getLineShaderProgram();
}

void Line::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Line::unbindVBO() {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
}

void Line::draw() {
    glEnable(GL_LINE_SMOOTH);
    glUseProgram(this->lineShaderProgram);
    glLineWidth(2.0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_LINES, 0, this->count * 2);
    
    glDisable(GL_LINE_SMOOTH);
}

void Line::tick() {
    int THREADS_PER_BLOCK = 1024;
    
    int threads = this->count % THREADS_PER_BLOCK;
    int blocks = (this->count + threads) / threads;
    
    moveLines<<<blocks, threads>>>(this->dptr, this->count);
}