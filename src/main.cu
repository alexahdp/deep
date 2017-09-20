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
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

const GLuint WIDTH = 800, HEIGHT = 600;

GLFWwindow* window;
GLuint lineShaderProgram;
GLuint pointShaderProgram;

typedef struct Point {
    float3 pos;
    //std::vector<float> pos;
    //std::vector<float> vel;
};

Point *point;

Point *dptr = NULL;

struct Line {
    std::vector<int> ft;
    std::vector<float> pos;
} typedef Line;
Line line;


__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__global__ void simple_vbo_kernel(Point *point) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //pos[i] = pos[i] + make_float3(0.001, 0.001, 0);
    point[i].pos = point[i].pos + make_float3(0.001, 0.001, 0);
}

void launch_kernel(Point *pos) {
    int p = 256;
    int q = 1;
    simple_vbo_kernel<<<1, 3 >>>(pos);
}

void createVBO(GLuint *vbo, cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, int size, void **dptr) {
    cudaGraphicsMapResources(1, vbo_res, 0);
    
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(dptr, &num_bytes, *vbo_res));
}

void deleteVBO(cudaGraphicsResource **vbo_res) {
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_res, 0));
}

void runCuda(Point *dptr) {
    launch_kernel(dptr);
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    std::cout << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    // point.pos.push_back(0.4);
    // point.pos.push_back(0.4);
    // point.pos.push_back(0);
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
    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
    
    GLenum glewinit = glewInit();
    if (glewinit != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << glewGetErrorString(glewinit) << std::endl;
        return -1;
    }

    // Define the viewport dimensions
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    
    return 0;
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
    
    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, line.pos.size() / 3);
    
    glDisable(GL_LINE_SMOOTH);
}

void drawPoints() {
    glUseProgram(pointShaderProgram);
    glPointSize(10.0);
    //glVertexPointer(2, GL_FLOAT, 0, pointVertex);
    
    glDrawArrays(GL_POINTS, 0, (sizeof(Point) / sizeof(float)) / 3);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
}


int main() {
    
    int res = init();
    if (res != 0) {
        return 1;
    }
    
    lineShaderProgram = getLineShaderProgram();
    pointShaderProgram = getPointShaderProgram();
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, 0, 1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    srand(time(NULL));
    
    point = (Point*)malloc(sizeof(Point));
    point[0].pos = {0.1, 0.1, 0};
    
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    
    cudaMalloc((void **) &dptr, sizeof(Point));
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point), 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard));
    
    createVBO(&VBO, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard, sizeof(Point), (void **)&dptr);
    cudaMemcpy((void *)&dptr, point, sizeof(Point), cudaMemcpyHostToDevice);
    deleteVBO(&cuda_vbo_resource);
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point), point, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    
    int fpsCounter = 0;
    clock_t current_ticks, delta_ticks, fps = 0;
    while (!glfwWindowShouldClose(window)) {
        current_ticks = clock();
        glfwPollEvents();
        glClearColor(0, 0, 0, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        //glEnable(GL_DEPTH_TEST);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableVertexAttribArray(0);
        
        //glBufferData(GL_ARRAY_BUFFER, line.pos.size() * sizeof(float), line.pos.data(), GL_DYNAMIC_DRAW);
        //drawLines();
        
        //glBufferData(GL_ARRAY_BUFFER, point.pos.size() * sizeof(float), point.pos.data(), GL_DYNAMIC_DRAW);
        
        drawPoints();
        //glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArray(0);
        
        glfwSwapBuffers(window);
        
        createVBO(&VBO, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard, sizeof(Point), (void **)&dptr);
        runCuda(dptr);
        deleteVBO(&cuda_vbo_resource);
        
        // FPS
        delta_ticks = clock() - current_ticks;
        if (delta_ticks > 0) {
            fps = CLOCKS_PER_SEC / delta_ticks;
        }
        
        if (fpsCounter % 100 == 0) {
            std::cout << "fps: " << fps << std::endl;
        }
        
        fpsCounter++;
        if (fpsCounter > 1000000) fpsCounter = 0;
    }

    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    
    cudaDeviceReset();
    
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    return 0;
}