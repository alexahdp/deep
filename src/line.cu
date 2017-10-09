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

int LineStructSize = (int)sizeof(LineStruct);

__global__ void moveLines(LineStruct* line, LineFTStruct* ft, PointStruct* point, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    
    line[i].pos.f = point[ ft[i].f ].pos;
    line[i].pos.t = point[ ft[i].t ].pos;
}

Line::Line(int _count) {
    this->count = _count;
    this->size = LineStructSize * _count;
    this->data = (LineStruct*)malloc(this->size);
    this->ft = (LineFTStruct*)malloc(sizeof(LineFTStruct) * _count);
    
    this->data[0].pos.f = {0, 0, 0};
    this->data[0].pos.t = {0.5, 0.5, 0};
    
    this->ft[0].f = 0;
    this->ft[0].t = 1;
    
    this->ftdptr = NULL;
    checkCudaErrors(cudaMalloc((void **)&this->ftdptr, (int)sizeof(LineFTStruct) * _count));
    checkCudaErrors(cudaMemcpy(this->ftdptr, this->ft, sizeof(LineFTStruct) * _count, cudaMemcpyHostToDevice));
    
    // for (int i = 0; i < this->count; i++) {
    //     this->data[i].pos.f = {trand(), trand(), 0};
    //     this->data[i].pos.t = {trand(), trand(), 0};
    // }
    
    glGenBuffers(1, &this->VBO);
    
    this->dptr = NULL;
    this->lineShaderProgram = getLineShaderProgram();
    
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, this->size, this->data, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&this->cuda_vbo_resource, this->VBO, cudaGraphicsMapFlagsNone));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Line::add(int f, int t) {
    this->count++;
    
    this->d2h();
    
    // а вот тут мне уже надо работать не с this->data, 
    // а с this->ft
    
    this->h2d();
}

void Line::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Line::unbindVBO() {
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

void Line::draw() {
    glEnable(GL_LINE_SMOOTH);
    glUseProgram(this->lineShaderProgram);
    glLineWidth(2.0);
    
    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, LineStructSize, (GLvoid *)0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    
    //glVertexPointer(3, GL_FLOAT, LineStructSize, 0);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    //glInterleavedArrays(GL_V2F, 0, NULL);
    
    glDrawArrays(GL_LINES, 0, this->count * 2);
    
    glDisable(GL_LINE_SMOOTH);
}

void Line::tick(Point* p1) {
    int THREADS_PER_BLOCK = 1024;
    
    int threads = this->count % THREADS_PER_BLOCK;
    int blocks = (this->count + threads) / threads;
    
    moveLines<<<blocks, threads>>>(this->dptr, this->ftdptr, p1->dptr, this->count);
}