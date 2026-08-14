#include "mygrad/threadPool.hpp"

#include <cstdlib>
#include <iostream>

namespace mygrad {

static constexpr size_t DEFAULT_POOL_SIZE = 8;

// MYGRAD_NUM_THREADS is read here, in the constructor, rather than through a
// setter: the pool is a lazily constructed singleton, so a setter would only
// take effect if it ran before the first push, and nothing could enforce that.
// Anything unusable falls back to the automatic size instead of throwing —
// this runs inside a static initialiser, a bad place to throw from.
static size_t poolSizeFromEnvironment() {
    const char* requestedThreads = std::getenv("MYGRAD_NUM_THREADS");
    if (requestedThreads and *requestedThreads) {
        char* firstUnparsedCharacter = nullptr;
        const unsigned long requested = std::strtoul(requestedThreads, &firstUnparsedCharacter, 10);
        if (*firstUnparsedCharacter == '\0' and requested > 0) { return requested; }

        std::cerr << "mygrad: MYGRAD_NUM_THREADS=\"" << requestedThreads
                  << "\" is not a positive number, sizing the thread pool automatically\n";
    }

    const size_t detectedCores = std::thread::hardware_concurrency();
    return detectedCores ? detectedCores : DEFAULT_POOL_SIZE;
}

ThreadPool::ThreadPool() :
    threads(),
    jobs(), jobsMutex(), jobsAvailable(), jobsRemaining(0) {
        const size_t poolSize = poolSizeFromEnvironment();
        threads.reserve(poolSize);
        for (size_t i = 0; i < poolSize; i++) {
            threads.emplace_back(std::thread(&threadLoop));
        }
    }

ThreadPool::~ThreadPool() {
    for (size_t i = 0; i < threads.size(); i++) {
        terminate = true;
        jobsAvailable.notify_all();
        threads[i].join();
    }
}

ThreadPool& ThreadPool::get() {
    static ThreadPool instance;
    return instance;
}

void ThreadPool::push(std::function<void()> job) {
    ThreadPool& pool = get();
    std::lock_guard lock (pool.jobsMutex);
    pool.jobs.push(job);
    pool.jobsRemaining++;
    pool.jobsAvailable.notify_one();
}


void ThreadPool::threadLoop() {
    ThreadPool& pool = get();
    while (true) {
        std::function<void()> job; 
        {
            std::unique_lock<std::mutex> lock(pool.jobsMutex);
            pool.jobsAvailable.wait(lock, [&pool] { return !pool.jobs.empty() or pool.terminate; });

            if (pool.jobs.empty() and pool.terminate) { return; }

            job = pool.jobs.front();
            pool.jobs.pop();
        }
        job();

        if (--pool.jobsRemaining == 0) {
            std::unique_lock lock(pool.doneMutex);
            pool.allDone.notify_all();
        }
    }
}

void ThreadPool::waitUntilDone() {
    ThreadPool& pool = get();
    std::unique_lock lock(pool.doneMutex);
    pool.allDone.wait(lock, [&pool] {
        return pool.jobsRemaining == 0;
    });
}

} // namespace mygrad

