cmake_minimum_required(VERSION 3.12)
project(continuous_hmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS USING_CMAKE=1)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS})

set(LIB_HEADERS
  lib/hmm.hpp
  lib/gmm.hpp
  lib/mvn.hpp
  lib/aliases.hpp
)

set(LIB_SOURCES
  lib/hmm.cpp
  lib/gmm.cpp
  lib/mvn.cpp
)

add_library(continuous_hmm ${LIB_HEADERS} ${LIB_SOURCES})
target_link_libraries(continuous_hmm Eigen3::Eigen)

#set(HEADERS
#  main.hpp
#)

#set(SOURCES
#  main.cpp
#)

#add_executable(demo ${HEADERS} ${SOURCES})
#target_link_libraries(demo gmm_hmm)
