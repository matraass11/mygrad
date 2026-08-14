include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.15.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# sources listed explicitly, like the library's
add_executable(mygradTests
    smoke.cpp
)

target_link_libraries(mygradTests PRIVATE mygrad Catch2::Catch2WithMain)

# Two ctest invocations of the same binary at different thread counts, per
# ticket 003: the pool is sized once per process from MYGRAD_NUM_THREADS
# (ticket 013), so there is no mid-process resize to test around. An empty
# value means "auto" and shields the N-thread run from an inherited setting.
add_test(NAME mygrad-1thread COMMAND mygradTests)
set_tests_properties(mygrad-1thread PROPERTIES ENVIRONMENT "MYGRAD_NUM_THREADS=1")

add_test(NAME mygrad-nthreads COMMAND mygradTests)
set_tests_properties(mygrad-nthreads PROPERTIES ENVIRONMENT "MYGRAD_NUM_THREADS=")
