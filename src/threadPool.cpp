#include "mygrad/threadPool.hpp"

#include <iostream>

namespace mygrad {

static constexpr size_t DEFAULT_POOL_SIZE = 8;

ThreadPool::ThreadPool() : 
    threads(),
    jobs(), jobsMutex(), jobsAvailable(), jobsRemaining(0) {
        const size_t poolSize = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : DEFAULT_POOL_SIZE;
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

// void ThreadPool::waitUntilFinished() {

// }

} // namespace mygrad

