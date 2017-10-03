//
//  Полезные ссылки
//    https://solarianprogrammer.com/2013/05/13/opengl-101-drawing-primitives/
//    http://vbomesh.blogspot.ru/2012/02/vbo-opengl.html
//    https://www.khronos.org/opengl/wiki/VBO_-_just_examples
//    http://pmg.org.ru/nehe/
//    http://www.codenet.ru/progr/opengl/


// #define GLEW_STATIC
#include <stdio.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include "lineShader.hpp"
//#include "pointShader.hpp"
#include "point.cuh"

// vbo variables
// GLuint VBO, VAO, VBO2;
GLuint VBO, VAO, VBO2;
GLuint VBOS[2];

struct cudaGraphicsResource *cuda_vbo_resource2;

const GLuint WIDTH = 800, HEIGHT = 600;

GLFWwindow* window;
GLuint lineShaderProgram;

struct pos2 {
    float3 f;
    float3 t;
};
struct Line {
    pos2 pos;
} typedef Line;
Line *line;
int linesCount = 1;
int lineSize = sizeof(Line) * linesCount;
Line *ldptr = NULL;



void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    std::cout << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}


void drawLines() {
    glEnable(GL_LINE_SMOOTH);
    glUseProgram(lineShaderProgram);
    //glVertexPointer(3, GL_FLOAT, 0, NULL);
    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, (lineSize / sizeof(float)) / 3);
    
    glDisable(GL_LINE_SMOOTH);
}


int init() {
    std::cout << "Starting GLFW context, OpenGL 4.5" << std::endl;
    
    glfwInit();
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    
    // Create a GLFWwindow object that we can use for GLFW's functions
    window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    
    glewExperimental = GL_TRUE;
    
    GLenum glewinit = glewInit();
    if (glewinit != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << glewGetErrorString(glewinit) << std::endl;
        return -1;
    }
    
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    
    lineShaderProgram = getLineShaderProgram();
    
    return 0;
}

void loop(Ppoint* p1) {
    glfwPollEvents();
    glClearColor(0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    //glEnable(GL_DEPTH_TEST);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, p1->VBO);
    //glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    p1->draw();
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    drawLines();
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    //glDisable(GL_DEPTH_TEST);
    glDisableClientState(GL_VERTEX_ARRAY);
    
    glfwSwapBuffers(window);
    
    p1->bindVBO();
    p1->tick();
    //launch_kernel(p1->dptr);
    p1->unbindVBO();
}


int main() {
    if (init() != 0) return 1;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, 0, 1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    Ppoint* p1 = new Ppoint(3);
    
    line = (Line*)malloc(lineSize);
    line[0].pos.f = {-0.5, 0, 0};
    line[0].pos.t = {0, -0.5, 0};
    
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    
    glGenBuffers(2, VBOS);
    p1->VBO = VBOS[0];
    VBO2 = VBOS[1];
    
    glBindBuffer(GL_ARRAY_BUFFER, p1->VBO);
    glBufferData(GL_ARRAY_BUFFER, p1->size, p1->data, GL_DYNAMIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, pointSize, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&p1->cuda_vbo_resource, p1->VBO, cudaGraphicsMapFlagsNone));
    // bindVBO(&cuda_vbo_resource, (void **)&dptr);
    // cudaMemcpy(dptr, point, pointSize, cudaMemcpyHostToDevice);
    // unbindVBO(&cuda_vbo_resource);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, lineSize, line, GL_DYNAMIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, lineSize, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource2, VBO2, cudaGraphicsMapFlagsNone));
    // bindVBO(&cuda_vbo_resource2, (void **)&ldptr);
    // cudaMemcpy(ldptr, line, lineSize, cudaMemcpyHostToDevice);
    // unbindVBO(&cuda_vbo_resource2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    while (!glfwWindowShouldClose(window)) loop(p1);
    
    checkCudaErrors(cudaGraphicsUnregisterResource(p1->cuda_vbo_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource2));
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &p1->VBO);
    glDeleteBuffers(1, &VBO2);
    
    cudaFree(p1->dptr);
    cudaFree(ldptr);
    
    cudaDeviceReset();
    glfwTerminate();
    
    return 0;
}