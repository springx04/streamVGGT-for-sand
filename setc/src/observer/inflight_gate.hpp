#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace omnivggt::observer {

class InFlightGate {
public:
    static constexpr std::size_t MAX_INFLIGHT_GROUPS = 3U;

    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return inflight_ < MAX_INFLIGHT_GROUPS || closed_; });
        if (!closed_) ++inflight_;
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        --inflight_;
        condition_.notify_one();
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t inflight_ = 0U;
    bool closed_ = false;
};

}  // namespace omnivggt::observer
