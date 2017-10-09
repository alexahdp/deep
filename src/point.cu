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

int PointStructSize = (int)sizeof(PointStruct);

Point::Point(int _count){
    this->COUNT = 10000;
    this->SIZE = PointStructSize * COUNT;
    
    this->count = _count;
    
    this->data = (PointStruct*)malloc(this->SIZE);
    
    for (int i = 0; i < this->count; i++) {
        this->data[i].pos = {trand(), trand(), 0};
        this->data[i].vel = {trand() / 500.0f, trand() / 500.0f, 0};
    }
    
    glGenBuffers(1, &this->VBO);
    
    this->dptr = NULL;
    
    this->pointShaderProgram = getPointShaderProgram();
    
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, this->SIZE, this->data, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&this->cuda_vbo_resource, this->VBO, cudaGraphicsMapFlagsNone));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int Point::size() {
    return PointStructSize * this->count;
}

void Point::add(float3 pos, float3 vel) {
    this->count++;
    
    this->d2h();
    this->data[this->count - 1].pos = pos;
    this->data[this->count - 1].vel = vel;
    this->h2d();
}

void Point::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Point::unbindVBO() {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
}

void Point::d2h() {
    this->bindVBO();
    
    checkCudaErrors(cudaMemcpy((void *)this->data, this->dptr, this->size(), cudaMemcpyDeviceToHost));
    this->unbindVBO();
}

void Point::h2d() {
    this->bindVBO();
    checkCudaErrors(cudaMemcpy(this->dptr, (void *)this->data, this->size(), cudaMemcpyHostToDevice));
    this->unbindVBO();
}

void Point::draw() {
    glUseProgram(this->pointShaderProgram);
    glPointSize(10.0);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, PointStructSize, (GLvoid *)0);
    glDrawArrays(GL_POINTS, 0, this->count);
}

void Point::tick() {
    int THREADS_PER_BLOCK = 1024;
    
    int threads = this->count % THREADS_PER_BLOCK;
    int blocks = (this->count + threads) / threads;
    
    simple_vbo_kernel<<<blocks, threads>>>(this->dptr, this->count);
}
