#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>

namespace mygrad {

class ThreadPool {
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
    std::mutex jobsMutex;
    std::condition_variable jobsAvailable;

    std::mutex doneMutex;
    std::condition_variable allDone;
    std::atomic<size_t> jobsRemaining;

    bool terminate = false;
    ThreadPool();
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    static ThreadPool& get();

    static void threadLoop();

public:
    static void push(std::function<void()>);
    static void waitUntilDone();
    static bool busy();
    static size_t size() { return get().threads.size(); };

};

} // namespace mygrad