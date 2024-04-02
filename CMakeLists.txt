cmake_minimum_required(VERSION 3.25)
project(LearnOpenCV)

set(CMAKE_CXX_STANDARD 11)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LearnOpenCV
#        main.cpp
#        assign1.cpp
#        Quiz1/Quiz1.cpp
        Quiz2/Quiz2.cpp
        )



target_link_libraries(LearnOpenCV ${OpenCV_LIBS})
