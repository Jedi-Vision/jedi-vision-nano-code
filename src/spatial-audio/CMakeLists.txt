cmake_minimum_required(VERSION 4.1)

project(spatial-audio-demo)

# Export compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(fmt CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)
find_package(pffft CONFIG REQUIRED)
find_package(mysofa CONFIG REQUIRED)
find_package(steam-audio CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(spatial-audio-demo spatial-audio-demo.cpp)

target_link_libraries(spatial-audio-demo PRIVATE fmt::fmt unofficial::steam-audio::phonon)

add_executable(object-spatial-audio object-spatial-audio.cpp)

target_link_libraries(object-spatial-audio PRIVATE 
    fmt::fmt 
    unofficial::steam-audio::phonon
    nlohmann_json::nlohmann_json
)