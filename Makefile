MAKEFLAGS=-r
CC=g++
CFLAGS=-Wall -Wextra -std=c++11
SRC_DIR=src
BIN_DIR=./bin
BUILD_DIR=./build
TARGET=bin/main
OBJECTS=$(BUILD_DIR)/%.o
MKDIR_P=mkdir -p

$(TARGET): $(OBJECTS)
	$(MKDIR_P) bin
	$(CC) $(OBJECTS) $(CFLAGS) -o $@ \
	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm

$(OBJECTS): $(SRC_DIR)/*.cpp
	$(MKDIR_P) build
	@$(CC) \
	-I /usr/include/GL -I /usr/local/include \
	$(CFLAGS) -c $< -o $@

clean:
	rm -f build/*
	rm -f bin/*