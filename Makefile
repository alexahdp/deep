MAKEFLAGS=-r
# CC=g++
CC=nvcc
#CFLAGS=-Wall -Wextra -std=c++11
CFLAGS=-std=c++11
SRC_DIR=src
BIN_DIR=./bin
BUILD_DIR=./build
TARGET=bin/main
#TARGET=bin/simpleGL
OBJECTS=$(BUILD_DIR)/%.o
MKDIR_P=mkdir -p

$(TARGET): $(OBJECTS)
	$(MKDIR_P) bin
	$(CC) $(OBJECTS) $(CFLAGS) -o $@ \
	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm -lglut

#$(OBJECTS): $(SRC_DIR)/simpleGL.cu
$(OBJECTS): $(SRC_DIR)/*.cu
	$(MKDIR_P) build
	@$(CC) \
	-I /usr/include/GL -I /usr/local/include -I ./inc \
	$(CFLAGS) -c $< -o $@

clean:
	rm -f build/*
	rm -f bin/*