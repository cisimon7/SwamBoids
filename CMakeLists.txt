cmake_minimum_required(VERSION 3.21)
project(SwamBoids)

set(CMAKE_CXX_STANDARD 23)
set(BOID_LIB Boids)

set(PYMODULE ${PROJECT_SOURCE_DIR}/PyModule)
set(LIBRARY_DIR ${PYMODULE}/venv/lib/python3.9/site-packages)

set(DEP_DIR ${PROJECT_SOURCE_DIR}/dependencies)
set(BOIDS_DIR ${PROJECT_SOURCE_DIR}/Boids)
set(PYBIND_DIR ${DEP_DIR}/pybind11-master)
set(SFML_DIR ${DEP_DIR}/SFML)
set(OPEN_MP_DIR ${DEP_DIR}/openMP)
set(OPEN_MP_LIB ${OPEN_MP_DIR}/lib/libomp.dylib)

add_subdirectory(${SFML_DIR})
add_subdirectory(${PYBIND_DIR})

file(GLOB BOIDS_SRC CONFIGURE_DEPENDS "${BOIDS_DIR}/src/*.cpp")
file(GLOB INCLUDE_DIRS CONFIGURE_DEPENDS "${SFML_DIR}/include" "${OPEN_MP_DIR}/include" "${BOIDS_DIR}/include")
add_library(${BOID_LIB} ${BOIDS_SRC} ${BOIDS_DIR}/include)
target_include_directories(${BOID_LIB} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${BOID_LIB} PRIVATE pybind11::embed sfml-graphics sfml-window sfml-system ${OPEN_MP_LIB})

# Flag to use OpenMP
SET(OPENMP_COMPILE_FLAGS "-Xpreprocessor -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENMP_LINK_FLAGS}")


add_executable(${PROJECT_NAME} main.cpp cxxopts.hpp)
target_link_libraries(${PROJECT_NAME} ${BOID_LIB})

pybind11_add_module(
        SimulationModule
        ${BOIDS_DIR}/pybinds/SimulationModule.cpp
)
target_include_directories(SimulationModule PUBLIC ${INCLUDE_DIRS})
target_link_libraries(SimulationModule PUBLIC ${BOID_LIB} sfml-graphics sfml-window sfml-system ${OPEN_MP_LIB})
set_target_properties(SimulationModule PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_DIR}/bindings)
