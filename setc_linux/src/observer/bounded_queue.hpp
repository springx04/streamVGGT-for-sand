#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace omnivggt::observer {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(const std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0U) {
            throw std::invalid_argument("BoundedQueue capacity must be positive");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // Latest-frame priority: if full, remove the oldest waiting item and
    // return it to the caller so the frame can be recorded as Coalesced.
    std::optional<T> push_latest(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return std::nullopt;
        }
        std::optional<T> dropped;
        if (queue_.size() >= capacity_) {
            dropped = std::move(queue_.front());
            queue_.pop_front();
        }
        queue_.push_back(std::move(value));
        condition_.notify_one();
        return dropped;
    }

    bool push_wait(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        queue_.push_back(std::move(value));
        condition_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop_front();
        condition_.notify_all();
        return value;
    }

    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop_front();
        condition_.notify_all();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        condition_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<T> queue_;
    bool closed_ = false;
};

}  // namespace omnivggt::observer
