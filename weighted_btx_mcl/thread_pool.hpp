#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// The calling thread participates, so a pool of N uses N execution threads.
// Waiting callers execute queued jobs, including nested protocol parallel loops.
class ThreadPool {
    using Work = std::function<void()>;
    struct Job {
        std::atomic<size_t> remaining{0};
        std::mutex mutex;
        std::exception_ptr error;
    };
    size_t count_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Work> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
    bool take(Work& work) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        work = std::move(queue_.back());
        queue_.pop_back();
        return true;
    }
public:
    explicit ThreadPool(size_t count) : count_(count) {
        if (!count) throw std::invalid_argument("thread count must be positive");
        for (size_t i = 1; i < count; ++i) workers_.emplace_back([this] {
            for (;;) {
                Work work;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                    if (stopping_ && queue_.empty()) return;
                    work = std::move(queue_.back());
                    queue_.pop_back();
                }
                work();
            }
        });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
        if (!n) return;
        if (count_ == 1 || n == 1) {
            for (size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        auto job = std::make_shared<Job>();
        const size_t chunk = std::max<size_t>(1, (n + count_ * 4 - 1) / (count_ * 4));
        job->remaining = (n + chunk - 1) / chunk;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t begin = 0; begin < n; begin += chunk) {
                const size_t end = std::min(n, begin + chunk);
                queue_.emplace_back([&, job, begin, end] {
                    try { for (size_t i = begin; i < end; ++i) fn(i); }
                    catch (...) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        if (!job->error) job->error = std::current_exception();
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        job->remaining.fetch_sub(1, std::memory_order_release);
                    }
                    ready_.notify_all();
                });
            }
        }
        ready_.notify_all();
        while (job->remaining.load(std::memory_order_acquire)) {
            Work work;
            if (take(work)) work();
            else {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return !queue_.empty() || !job->remaining.load(std::memory_order_acquire); });
            }
        }
        if (job->error) std::rethrow_exception(job->error);
    }
};
