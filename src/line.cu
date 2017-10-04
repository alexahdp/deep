#include <stdio.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include "lineShader.hpp"
//#include "point.cuh"

struct pos2 {
    float3 f;
    float3 t;
};
struct LineStruct {
    pos2 pos;
};


class Line {
    public:
        int size;
        int count;
        GLuint lineShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        LineStruct* data;
        LineStruct *dptr;
        
        Line(int _count);
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick();
};

Line::Line(int _count) {
    this->count = _count;
    this->size = sizeof(LineStruct) * _count;
    this->data = (LineStruct*)malloc(this->size);
    
    this->data = (LineStruct*)malloc(this->size);
    this->data->pos.f = {-0.5, 0, 0};
    this->data->pos.t = {0, -0.5, 0};
    
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
    //glVertexPointer(3, GL_FLOAT, 0, NULL);
    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, (this->size / sizeof(float)) / 3);
    
    glDisable(GL_LINE_SMOOTH);
}
