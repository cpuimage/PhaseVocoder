cmake_minimum_required(VERSION 2.8)
project(phase_vocoder)

add_executable(phase_vocoder pv_example.c phase_vocoder.c)
target_link_libraries(phase_vocoder -lm)