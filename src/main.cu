//
//  Полезные ссылки
//    https://solarianprogrammer.com/2013/05/13/opengl-101-drawing-primitives/
//    http://vbomesh.blogspot.ru/2012/02/vbo-opengl.html
//    https://www.khronos.org/opengl/wiki/VBO_-_just_examples
//    http://pmg.org.ru/nehe/
//    http://www.codenet.ru/progr/opengl/


// #define GLEW_STATIC
#include <stdio.h>
#include <math.h>
#include <glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <ctime>

//#include "lineShader.hpp"
//#include "pointShader.hpp"
#include "point.cuh"
#include "line.cuh"
#include "lib.hpp"

GLuint VAO;
GLuint VBOS[2];

const GLuint WIDTH = 800, HEIGHT = 600;
GLFWwindow* window;

Point* p1;
Line* l1;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    
    std::cout << key << std::endl;
    std::cout << "===" << std::endl;
    
    if (action == GLFW_PRESS) {
        float3 pos = {trand(), trand(), 0};
        float3 vel = {trand(), trand(), 0};
        
        p1->add(pos, vel);
    }
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
    
    return 0;
}


void loop(Point* p1, Line* l1) {
    glfwPollEvents();
    glClearColor(0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    //glEnable(GL_DEPTH_TEST);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, p1->VBO);
    //glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)0);
    glEnableVertexAttribArray(0);
    p1->draw();
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, l1->VBO);
    glEnableVertexAttribArray(0);
    l1->draw();
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    
    glfwSwapBuffers(window);
    
    p1->bindVBO();
    p1->tick();
    p1->unbindVBO();
    
    l1->bindVBO();
    l1->tick();
    l1->unbindVBO();
}


int main() {
    
    std::srand(unsigned(std::time(0)));
    
    if (init() != 0) return 1;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, 0, 1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    p1 = new Point(2);
    l1 = new Line(1);
    
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    
    glGenBuffers(2, VBOS);
    p1->VBO = VBOS[0];
    l1->VBO = VBOS[1];
    
    glBindBuffer(GL_ARRAY_BUFFER, p1->VBO);
    glBufferData(GL_ARRAY_BUFFER, p1->SIZE, p1->data, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&p1->cuda_vbo_resource, p1->VBO, cudaGraphicsMapFlagsNone));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, l1->VBO);
    glBufferData(GL_ARRAY_BUFFER, l1->size, l1->data, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&l1->cuda_vbo_resource, l1->VBO, cudaGraphicsMapFlagsNone));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    while (!glfwWindowShouldClose(window)) loop(p1, l1);
    
    checkCudaErrors(cudaGraphicsUnregisterResource(p1->cuda_vbo_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(l1->cuda_vbo_resource));
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &p1->VBO);
    glDeleteBuffers(1, &l1->VBO);
    
    cudaFree(p1->dptr);
    cudaFree(l1->dptr);
    
    cudaDeviceReset();
    glfwTerminate();
    
    return 0;
}