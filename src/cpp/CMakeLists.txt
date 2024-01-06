cmake_minimum_required(VERSION 3.0.0)
project(Projectpybind11 VERSION 0.1.0)

# include(CTest)
# enable_testing()

find_package(pybind11 REQUIRED)

# Add -O3 to the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

pybind11_add_module(engine_ia node.cpp point.cpp road.cpp movable.cpp module.cpp)
target_link_libraries(engine_ia PRIVATE ${MY_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
