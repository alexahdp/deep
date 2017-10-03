#include <stdio.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include "pointShader.hpp"
//#include "point.cuh"

struct Point {
    float3 pos;
};


__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__global__ void simple_vbo_kernel(Point *point) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    point[i].pos = point[i].pos + make_float3(0.001, 0.001, 0);
}


class Ppoint {
    public:
        int size;
        int count;
        GLuint pointShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        Point* data;
        Ppoint(int _count);
        Point *dptr;
        
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick();
};

//int Ppoint::pointSize = sizeof(Point);

Ppoint::Ppoint(int _count) {
    this->count = _count;
    this->size = sizeof(Point) * _count;
    this->data = (Point*)malloc(this->size);
    
    this->data[0].pos = {0.1, 0.1, 0};
    this->data[1].pos = {0, 0.1, 0};
    this->data[2].pos = {0.1, 0, 0};
    
    this->dptr = NULL;
    
    this->pointShaderProgram = getPointShaderProgram();
}

void Ppoint::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Ppoint::unbindVBO() {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
}

void Ppoint::draw() {
    glUseProgram(this->pointShaderProgram);
    glPointSize(10.0);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    //glVertexPointer(3, GL_FLOAT, 0, dptr);
    
    //glDrawArrays(GL_POINTS, 0, (pointSize / sizeof(float)) / 3);
    glDrawArrays(GL_POINTS, 0, 3);
}

void Ppoint::tick() {
    //int blocks = sizeof(&point) / pointSize;
    simple_vbo_kernel<<<1, 9>>>(this->dptr);
}