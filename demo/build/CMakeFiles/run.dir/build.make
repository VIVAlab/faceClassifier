# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/binghao/faceClassifier/demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/binghao/faceClassifier/demo/build

# Include any dependencies generated for this target.
include CMakeFiles/run.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run.dir/flags.make

CMakeFiles/run.dir/model.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/model.c.o: ../model.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/model.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/model.c.o   -c /home/binghao/faceClassifier/demo/model.c

CMakeFiles/run.dir/model.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/model.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/model.c > CMakeFiles/run.dir/model.c.i

CMakeFiles/run.dir/model.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/model.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/model.c -o CMakeFiles/run.dir/model.c.s

CMakeFiles/run.dir/model.c.o.requires:
.PHONY : CMakeFiles/run.dir/model.c.o.requires

CMakeFiles/run.dir/model.c.o.provides: CMakeFiles/run.dir/model.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/model.c.o.provides.build
.PHONY : CMakeFiles/run.dir/model.c.o.provides

CMakeFiles/run.dir/model.c.o.provides.build: CMakeFiles/run.dir/model.c.o

CMakeFiles/run.dir/12Layer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/12Layer.c.o: ../12Layer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/12Layer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/12Layer.c.o   -c /home/binghao/faceClassifier/demo/12Layer.c

CMakeFiles/run.dir/12Layer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/12Layer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/12Layer.c > CMakeFiles/run.dir/12Layer.c.i

CMakeFiles/run.dir/12Layer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/12Layer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/12Layer.c -o CMakeFiles/run.dir/12Layer.c.s

CMakeFiles/run.dir/12Layer.c.o.requires:
.PHONY : CMakeFiles/run.dir/12Layer.c.o.requires

CMakeFiles/run.dir/12Layer.c.o.provides: CMakeFiles/run.dir/12Layer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/12Layer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/12Layer.c.o.provides

CMakeFiles/run.dir/12Layer.c.o.provides.build: CMakeFiles/run.dir/12Layer.c.o

CMakeFiles/run.dir/12CLayer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/12CLayer.c.o: ../12CLayer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/12CLayer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/12CLayer.c.o   -c /home/binghao/faceClassifier/demo/12CLayer.c

CMakeFiles/run.dir/12CLayer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/12CLayer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/12CLayer.c > CMakeFiles/run.dir/12CLayer.c.i

CMakeFiles/run.dir/12CLayer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/12CLayer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/12CLayer.c -o CMakeFiles/run.dir/12CLayer.c.s

CMakeFiles/run.dir/12CLayer.c.o.requires:
.PHONY : CMakeFiles/run.dir/12CLayer.c.o.requires

CMakeFiles/run.dir/12CLayer.c.o.provides: CMakeFiles/run.dir/12CLayer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/12CLayer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/12CLayer.c.o.provides

CMakeFiles/run.dir/12CLayer.c.o.provides.build: CMakeFiles/run.dir/12CLayer.c.o

CMakeFiles/run.dir/24Layer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/24Layer.c.o: ../24Layer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/24Layer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/24Layer.c.o   -c /home/binghao/faceClassifier/demo/24Layer.c

CMakeFiles/run.dir/24Layer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/24Layer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/24Layer.c > CMakeFiles/run.dir/24Layer.c.i

CMakeFiles/run.dir/24Layer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/24Layer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/24Layer.c -o CMakeFiles/run.dir/24Layer.c.s

CMakeFiles/run.dir/24Layer.c.o.requires:
.PHONY : CMakeFiles/run.dir/24Layer.c.o.requires

CMakeFiles/run.dir/24Layer.c.o.provides: CMakeFiles/run.dir/24Layer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/24Layer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/24Layer.c.o.provides

CMakeFiles/run.dir/24Layer.c.o.provides.build: CMakeFiles/run.dir/24Layer.c.o

CMakeFiles/run.dir/24CLayer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/24CLayer.c.o: ../24CLayer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/24CLayer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/24CLayer.c.o   -c /home/binghao/faceClassifier/demo/24CLayer.c

CMakeFiles/run.dir/24CLayer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/24CLayer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/24CLayer.c > CMakeFiles/run.dir/24CLayer.c.i

CMakeFiles/run.dir/24CLayer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/24CLayer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/24CLayer.c -o CMakeFiles/run.dir/24CLayer.c.s

CMakeFiles/run.dir/24CLayer.c.o.requires:
.PHONY : CMakeFiles/run.dir/24CLayer.c.o.requires

CMakeFiles/run.dir/24CLayer.c.o.provides: CMakeFiles/run.dir/24CLayer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/24CLayer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/24CLayer.c.o.provides

CMakeFiles/run.dir/24CLayer.c.o.provides.build: CMakeFiles/run.dir/24CLayer.c.o

CMakeFiles/run.dir/48Layer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/48Layer.c.o: ../48Layer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/48Layer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/48Layer.c.o   -c /home/binghao/faceClassifier/demo/48Layer.c

CMakeFiles/run.dir/48Layer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/48Layer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/48Layer.c > CMakeFiles/run.dir/48Layer.c.i

CMakeFiles/run.dir/48Layer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/48Layer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/48Layer.c -o CMakeFiles/run.dir/48Layer.c.s

CMakeFiles/run.dir/48Layer.c.o.requires:
.PHONY : CMakeFiles/run.dir/48Layer.c.o.requires

CMakeFiles/run.dir/48Layer.c.o.provides: CMakeFiles/run.dir/48Layer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/48Layer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/48Layer.c.o.provides

CMakeFiles/run.dir/48Layer.c.o.provides.build: CMakeFiles/run.dir/48Layer.c.o

CMakeFiles/run.dir/48CLayer.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/48CLayer.c.o: ../48CLayer.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/48CLayer.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/48CLayer.c.o   -c /home/binghao/faceClassifier/demo/48CLayer.c

CMakeFiles/run.dir/48CLayer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/48CLayer.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/48CLayer.c > CMakeFiles/run.dir/48CLayer.c.i

CMakeFiles/run.dir/48CLayer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/48CLayer.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/48CLayer.c -o CMakeFiles/run.dir/48CLayer.c.s

CMakeFiles/run.dir/48CLayer.c.o.requires:
.PHONY : CMakeFiles/run.dir/48CLayer.c.o.requires

CMakeFiles/run.dir/48CLayer.c.o.provides: CMakeFiles/run.dir/48CLayer.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/48CLayer.c.o.provides.build
.PHONY : CMakeFiles/run.dir/48CLayer.c.o.provides

CMakeFiles/run.dir/48CLayer.c.o.provides.build: CMakeFiles/run.dir/48CLayer.c.o

CMakeFiles/run.dir/itos.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/itos.c.o: ../itos.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/itos.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/itos.c.o   -c /home/binghao/faceClassifier/demo/itos.c

CMakeFiles/run.dir/itos.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/itos.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/itos.c > CMakeFiles/run.dir/itos.c.i

CMakeFiles/run.dir/itos.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/itos.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/itos.c -o CMakeFiles/run.dir/itos.c.s

CMakeFiles/run.dir/itos.c.o.requires:
.PHONY : CMakeFiles/run.dir/itos.c.o.requires

CMakeFiles/run.dir/itos.c.o.provides: CMakeFiles/run.dir/itos.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/itos.c.o.provides.build
.PHONY : CMakeFiles/run.dir/itos.c.o.provides

CMakeFiles/run.dir/itos.c.o.provides.build: CMakeFiles/run.dir/itos.c.o

CMakeFiles/run.dir/doPyrDown.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/doPyrDown.c.o: ../doPyrDown.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/doPyrDown.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/doPyrDown.c.o   -c /home/binghao/faceClassifier/demo/doPyrDown.c

CMakeFiles/run.dir/doPyrDown.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/doPyrDown.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/doPyrDown.c > CMakeFiles/run.dir/doPyrDown.c.i

CMakeFiles/run.dir/doPyrDown.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/doPyrDown.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/doPyrDown.c -o CMakeFiles/run.dir/doPyrDown.c.s

CMakeFiles/run.dir/doPyrDown.c.o.requires:
.PHONY : CMakeFiles/run.dir/doPyrDown.c.o.requires

CMakeFiles/run.dir/doPyrDown.c.o.provides: CMakeFiles/run.dir/doPyrDown.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/doPyrDown.c.o.provides.build
.PHONY : CMakeFiles/run.dir/doPyrDown.c.o.provides

CMakeFiles/run.dir/doPyrDown.c.o.provides.build: CMakeFiles/run.dir/doPyrDown.c.o

CMakeFiles/run.dir/preprocess.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/preprocess.c.o: ../preprocess.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/preprocess.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/preprocess.c.o   -c /home/binghao/faceClassifier/demo/preprocess.c

CMakeFiles/run.dir/preprocess.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/preprocess.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/preprocess.c > CMakeFiles/run.dir/preprocess.c.i

CMakeFiles/run.dir/preprocess.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/preprocess.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/preprocess.c -o CMakeFiles/run.dir/preprocess.c.s

CMakeFiles/run.dir/preprocess.c.o.requires:
.PHONY : CMakeFiles/run.dir/preprocess.c.o.requires

CMakeFiles/run.dir/preprocess.c.o.provides: CMakeFiles/run.dir/preprocess.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/preprocess.c.o.provides.build
.PHONY : CMakeFiles/run.dir/preprocess.c.o.provides

CMakeFiles/run.dir/preprocess.c.o.provides.build: CMakeFiles/run.dir/preprocess.c.o

CMakeFiles/run.dir/freeArray.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/freeArray.c.o: ../freeArray.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/freeArray.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/freeArray.c.o   -c /home/binghao/faceClassifier/demo/freeArray.c

CMakeFiles/run.dir/freeArray.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/freeArray.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/freeArray.c > CMakeFiles/run.dir/freeArray.c.i

CMakeFiles/run.dir/freeArray.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/freeArray.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/freeArray.c -o CMakeFiles/run.dir/freeArray.c.s

CMakeFiles/run.dir/freeArray.c.o.requires:
.PHONY : CMakeFiles/run.dir/freeArray.c.o.requires

CMakeFiles/run.dir/freeArray.c.o.provides: CMakeFiles/run.dir/freeArray.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/freeArray.c.o.provides.build
.PHONY : CMakeFiles/run.dir/freeArray.c.o.provides

CMakeFiles/run.dir/freeArray.c.o.provides.build: CMakeFiles/run.dir/freeArray.c.o

CMakeFiles/run.dir/multiplyByElement.c.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/multiplyByElement.c.o: ../multiplyByElement.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/binghao/faceClassifier/demo/build/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/run.dir/multiplyByElement.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/run.dir/multiplyByElement.c.o   -c /home/binghao/faceClassifier/demo/multiplyByElement.c

CMakeFiles/run.dir/multiplyByElement.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run.dir/multiplyByElement.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/binghao/faceClassifier/demo/multiplyByElement.c > CMakeFiles/run.dir/multiplyByElement.c.i

CMakeFiles/run.dir/multiplyByElement.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run.dir/multiplyByElement.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/binghao/faceClassifier/demo/multiplyByElement.c -o CMakeFiles/run.dir/multiplyByElement.c.s

CMakeFiles/run.dir/multiplyByElement.c.o.requires:
.PHONY : CMakeFiles/run.dir/multiplyByElement.c.o.requires

CMakeFiles/run.dir/multiplyByElement.c.o.provides: CMakeFiles/run.dir/multiplyByElement.c.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/multiplyByElement.c.o.provides.build
.PHONY : CMakeFiles/run.dir/multiplyByElement.c.o.provides

CMakeFiles/run.dir/multiplyByElement.c.o.provides.build: CMakeFiles/run.dir/multiplyByElement.c.o

# Object files for target run
run_OBJECTS = \
"CMakeFiles/run.dir/model.c.o" \
"CMakeFiles/run.dir/12Layer.c.o" \
"CMakeFiles/run.dir/12CLayer.c.o" \
"CMakeFiles/run.dir/24Layer.c.o" \
"CMakeFiles/run.dir/24CLayer.c.o" \
"CMakeFiles/run.dir/48Layer.c.o" \
"CMakeFiles/run.dir/48CLayer.c.o" \
"CMakeFiles/run.dir/itos.c.o" \
"CMakeFiles/run.dir/doPyrDown.c.o" \
"CMakeFiles/run.dir/preprocess.c.o" \
"CMakeFiles/run.dir/freeArray.c.o" \
"CMakeFiles/run.dir/multiplyByElement.c.o"

# External object files for target run
run_EXTERNAL_OBJECTS =

run: CMakeFiles/run.dir/model.c.o
run: CMakeFiles/run.dir/12Layer.c.o
run: CMakeFiles/run.dir/12CLayer.c.o
run: CMakeFiles/run.dir/24Layer.c.o
run: CMakeFiles/run.dir/24CLayer.c.o
run: CMakeFiles/run.dir/48Layer.c.o
run: CMakeFiles/run.dir/48CLayer.c.o
run: CMakeFiles/run.dir/itos.c.o
run: CMakeFiles/run.dir/doPyrDown.c.o
run: CMakeFiles/run.dir/preprocess.c.o
run: CMakeFiles/run.dir/freeArray.c.o
run: CMakeFiles/run.dir/multiplyByElement.c.o
run: CMakeFiles/run.dir/build.make
run: /usr/local/lib/libopencv_viz.so.2.4.9
run: /usr/local/lib/libopencv_videostab.so.2.4.9
run: /usr/local/lib/libopencv_video.so.2.4.9
run: /usr/local/lib/libopencv_ts.a
run: /usr/local/lib/libopencv_superres.so.2.4.9
run: /usr/local/lib/libopencv_stitching.so.2.4.9
run: /usr/local/lib/libopencv_photo.so.2.4.9
run: /usr/local/lib/libopencv_ocl.so.2.4.9
run: /usr/local/lib/libopencv_objdetect.so.2.4.9
run: /usr/local/lib/libopencv_nonfree.so.2.4.9
run: /usr/local/lib/libopencv_ml.so.2.4.9
run: /usr/local/lib/libopencv_legacy.so.2.4.9
run: /usr/local/lib/libopencv_imgproc.so.2.4.9
run: /usr/local/lib/libopencv_highgui.so.2.4.9
run: /usr/local/lib/libopencv_gpu.so.2.4.9
run: /usr/local/lib/libopencv_flann.so.2.4.9
run: /usr/local/lib/libopencv_features2d.so.2.4.9
run: /usr/local/lib/libopencv_core.so.2.4.9
run: /usr/local/lib/libopencv_contrib.so.2.4.9
run: /usr/local/lib/libopencv_calib3d.so.2.4.9
run: /usr/lib/x86_64-linux-gnu/libGLU.so
run: /usr/lib/x86_64-linux-gnu/libGL.so
run: /usr/lib/x86_64-linux-gnu/libSM.so
run: /usr/lib/x86_64-linux-gnu/libICE.so
run: /usr/lib/x86_64-linux-gnu/libX11.so
run: /usr/lib/x86_64-linux-gnu/libXext.so
run: /usr/local/lib/libopencv_nonfree.so.2.4.9
run: /usr/local/lib/libopencv_ocl.so.2.4.9
run: /usr/local/lib/libopencv_gpu.so.2.4.9
run: /usr/local/lib/libopencv_photo.so.2.4.9
run: /usr/local/lib/libopencv_objdetect.so.2.4.9
run: /usr/local/lib/libopencv_legacy.so.2.4.9
run: /usr/local/lib/libopencv_video.so.2.4.9
run: /usr/local/lib/libopencv_ml.so.2.4.9
run: /usr/local/lib/libopencv_calib3d.so.2.4.9
run: /usr/local/lib/libopencv_features2d.so.2.4.9
run: /usr/local/lib/libopencv_highgui.so.2.4.9
run: /usr/local/lib/libopencv_imgproc.so.2.4.9
run: /usr/local/lib/libopencv_flann.so.2.4.9
run: /usr/local/lib/libopencv_core.so.2.4.9
run: CMakeFiles/run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable run"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run.dir/build: run
.PHONY : CMakeFiles/run.dir/build

CMakeFiles/run.dir/requires: CMakeFiles/run.dir/model.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/12Layer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/12CLayer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/24Layer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/24CLayer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/48Layer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/48CLayer.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/itos.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/doPyrDown.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/preprocess.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/freeArray.c.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/multiplyByElement.c.o.requires
.PHONY : CMakeFiles/run.dir/requires

CMakeFiles/run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run.dir/clean

CMakeFiles/run.dir/depend:
	cd /home/binghao/faceClassifier/demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/binghao/faceClassifier/demo /home/binghao/faceClassifier/demo /home/binghao/faceClassifier/demo/build /home/binghao/faceClassifier/demo/build /home/binghao/faceClassifier/demo/build/CMakeFiles/run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run.dir/depend

