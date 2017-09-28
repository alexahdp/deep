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

// vbo variables
// GLuint VBO, VAO, VBO2;
GLuint VBO, VAO, VBO2;
GLuint VBOS[2];

struct cudaGraphicsResource *cuda_vbo_resource;
struct cudaGraphicsResource *cuda_vbo_resource2;
void *d_vbo_buffer = NULL;

const GLuint WIDTH = 800, HEIGHT = 600;

GLFWwindow* window;
GLuint lineShaderProgram;
GLuint pointShaderProgram;

struct Point {
    float3 pos;
    //std::vector<float> pos;
    //std::vector<float> vel;
};
Point *point;
int pointsCount = 3;
int pointSize = sizeof(Point) * pointsCount;
Point *dptr = NULL;

struct pos2 {
    float3 f;
    float3 t;
};
struct Line {
    //float pos[6];
    pos2 pos;
} typedef Line;
Line *line;
int linesCount = 1;
int lineSize = sizeof(Line) * linesCount;
Line *ldptr = NULL;



__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__global__ void simple_vbo_kernel(Point *point) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    point[i].pos = point[i].pos + make_float3(0.001, 0.001, 0);
}

void launch_kernel(Point *point) {
    //int blocks = sizeof(&point) / pointSize;
    simple_vbo_kernel<<<1, 9>>>(point);
}

void bindVBO(cudaGraphicsResource **vbo_res, void **dptr) {
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_res, 0));
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(dptr, &num_bytes, *vbo_res));
}

void unbindVBO(cudaGraphicsResource **vbo_res) {
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_res, 0));
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    std::cout << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}


GLuint getLineShaderProgram() {
    // Shaders
    const GLchar* vertexShaderSource = "#version 450 core\n"
        "in vec3 vertex_position;\n"
        "void main(void) {\n"
            "gl_Position = vec4(vertex_position, 1.0);\n"
        "}\n\0";

    const GLchar* fragmentShaderSource = "#version 450 core\n"
        "void main(void) {\n"
          "gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "}\n\0";
    
    // Build and compile our shader program
    // Vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

GLuint getPointShaderProgram() {
    // Shaders
    const GLchar* vertexShaderSource = "#version 450 core\n"
        "in vec3 vertex_position;\n"
        "void main(void) {\n"
            "gl_Position = vec4(vertex_position, 1.0);\n"
            "gl_PointSize = 5.0;\n"
        "}\n\0";

    const GLchar* fragmentShaderSource = "#version 450 core\n"
        "void main(void) {\n"
        "  vec2 pos = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);\n"
        "  float mag = dot(pos, pos);\n"
          "if (mag > 1.0) discard;\n"
          "gl_FragColor = vec4(1.0, .0, .0, 1.0);\n"
        "}\n\0";
    
    // Build and compile our shader program
    // Vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

void drawLines() {
    glEnable(GL_LINE_SMOOTH);
    glUseProgram(lineShaderProgram);
    //glVertexPointer(3, GL_FLOAT, 0, NULL);
    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, (lineSize / sizeof(float)) / 3);
    
    glDisable(GL_LINE_SMOOTH);
}

void drawPoints() {
    glUseProgram(pointShaderProgram);
    glPointSize(10.0);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    //glVertexPointer(3, GL_FLOAT, 0, dptr);
    
    //glDrawArrays(GL_POINTS, 0, (pointSize / sizeof(float)) / 3);
    glDrawArrays(GL_POINTS, 0, 3);
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
    pointShaderProgram = getPointShaderProgram();
    
    return 0;
}


void loop() {
    glfwPollEvents();
    glClearColor(0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    //glEnable(GL_DEPTH_TEST);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    drawPoints();
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
    
    bindVBO(&cuda_vbo_resource, (void **)&dptr);
    launch_kernel(dptr);
    unbindVBO(&cuda_vbo_resource);
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
    
    point = (Point*)malloc(pointSize);
    point[0].pos = {0.1, 0.1, 0};
    point[1].pos = {0, 0.1, 0};
    point[2].pos = {0.1, 0, 0};
    
    line = (Line*)malloc(lineSize);
    line[0].pos.f = {-0.5, 0, 0};
    line[0].pos.t = {0, -0.5, 0};
    
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    
    glGenBuffers(2, VBOS);
    VBO = VBOS[0];
    VBO2 = VBOS[1];
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, pointSize, point, GL_DYNAMIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, pointSize, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone));
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
    
    while (!glfwWindowShouldClose(window)) loop();
    
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource2));
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    
    cudaFree(dptr);
    cudaFree(ldptr);
    
    cudaDeviceReset();
    glfwTerminate();
    
    return 0;
}