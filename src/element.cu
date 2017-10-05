#include <stdio.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include "element.cuh"

Element::Element() {
}

void Element::bindVBO() {
    checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&this->dptr, &num_bytes, this->cuda_vbo_resource));
}

void Element::unbindVBO() {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, 0));
}

// void Element::d2h() {
//     this->bindVBO();
    
//     checkCudaErrors(cudaMemcpy((void *)this->data, this->dptr, this->size(), cudaMemcpyDeviceToHost));
//     this->unbindVBO();
// }

// void Element::h2d() {
//     this->bindVBO();
//     checkCudaErrors(cudaMemcpy(this->dptr, (void *)this->data, this->size(), cudaMemcpyHostToDevice));
//     this->unbindVBO();
// }