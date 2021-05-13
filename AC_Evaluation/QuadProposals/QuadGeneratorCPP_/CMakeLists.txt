cmake_minimum_required(VERSION 2.8)                         
project( QuadGeneratorCPP )                                                               
find_package( OpenCV REQUIRED )   

file(GLOB SRC
    "*.h"
    "*.cpp"
)

include_directories($ENV{AC_DIR})
include_directories($ENV{EIGEN_DIR})
add_subdirectory(pybind11)
      
pybind11_add_module(QuadGeneratorCPP "./main.cpp")
target_link_libraries( QuadGeneratorCPP ${OpenCV_LIBS} )  


