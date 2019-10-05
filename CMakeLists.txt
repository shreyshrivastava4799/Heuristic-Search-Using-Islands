cmake_minimum_required(VERSION 2.8.3)
project(A*)

## C++11
set( CMAKE_CXX_STANDARD 14 )

# Find the package
find_package(OpenCV REQUIRED)

## Linking 
include_directories(include ${OpenCV_INCLUDE_DIRS} )

## Creating executable:
add_executable(A* A*.cpp)
target_link_libraries(A*  ${OpenCV_LIBS})


