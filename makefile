CXX := g++
AR := ar

OMPFLAGS := -fopenmp

CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic $(OMPFLAGS)
CPPFLAGS := -Iinclude
ARFLAGS := rcs

SRC_DIR := src
INC_DIR := include
EXAMPLE_DIR := examples
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib

LIB_NAME := librouting.a
LIB := $(LIB_DIR)/$(LIB_NAME)

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
SRC_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

EXAMPLE_SRCS := $(filter %.cpp,$(wildcard $(EXAMPLE_DIR)/*.cpp))
EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.cpp,$(BIN_DIR)/%,$(EXAMPLE_SRCS))

.PHONY: all lib examples clean run-saxpy

all: lib examples

lib: $(LIB)

examples: $(EXAMPLE_BINS)

$(LIB): $(SRC_OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(SRC_OBJS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/%: $(EXAMPLE_DIR)/%.cpp $(LIB) | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(OBJ_DIR): | $(BUILD_DIR)
	mkdir -p $@

$(BIN_DIR): | $(BUILD_DIR)
	mkdir -p $@

$(LIB_DIR): | $(BUILD_DIR)
	mkdir -p $@

run-saxpy: $(BIN_DIR)/saxpy
	./$(BIN_DIR)/saxpy

clean:
	rm -rf $(BUILD_DIR)
