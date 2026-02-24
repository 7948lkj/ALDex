#ifndef RETRAIN_MANAGER_H
#define RETRAIN_MANAGER_H

#include <Common.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

// 重训练任务结构体
struct RetrainTask {
    std::vector<uint64_t> next_addr_content;
    std::vector<kv_pair> chain_kvs;
};

class RetrainManager {
public:
    static RetrainManager& getInstance() {
        static RetrainManager instance;
        return instance;
    }

    void addTask(const RetrainTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(task);
        cv_.notify_one();
    }

    bool getTask(RetrainTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || !running_; });

        if (!running_ && queue_.empty()) {
            return false;
        }

        task = queue_.front();
        queue_.pop();
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        running_ = false;
        cv_.notify_all();
    }

private:
    RetrainManager() : running_(true) {}
    ~RetrainManager() = default;

    std::queue<RetrainTask> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool running_;
};

#endif // RETRAIN_MANAGER_H