#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "master_client.h"
#include "task_manager.h"
#include "thread_pool.h"
#include "types.h"

namespace mooncake {

// Forward declarations
class Client;

/**
 * @brief Executor for replica copy operations
 */
class ReplicaCopyExecutor {
public:
    ReplicaCopyExecutor(std::shared_ptr<Client> client, std::shared_ptr<MasterClient> master_client)
        : client_(client), master_client_(master_client) {}

    /**
     * @brief Execute a replica copy task
     * @param payload Task payload containing key and target segments
     * @return ErrorCode indicating success or failure
     */
    ErrorCode Execute(const ReplicaCopyPayload& payload);

private:
    std::shared_ptr<Client> client_;
    std::shared_ptr<MasterClient> master_client_;
};

/**
 * @brief Executor for replica move operations
 */
class ReplicaMoveExecutor {
public:
    ReplicaMoveExecutor(std::shared_ptr<Client> client, std::shared_ptr<MasterClient> master_client)
        : client_(client), master_client_(master_client) {}

    /**
     * @brief Execute a replica move task
     * @param payload Task payload containing key, source and target segments
     * @return ErrorCode indicating success or failure
     */
    ErrorCode Execute(const ReplicaMovePayload& payload);

private:
    std::shared_ptr<Client> client_;
    std::shared_ptr<MasterClient> master_client_;
};

/**
 * @brief Task executor with thread pool for async task execution
 */
class TaskExecutor {
public:
    TaskExecutor(std::shared_ptr<Client> client, std::shared_ptr<MasterClient> master_client, size_t thread_pool_size = 4)
        : client_(client), master_client_(master_client),
          thread_pool_(thread_pool_size),
          copy_executor_(client, master_client),
          move_executor_(client, master_client),
          running_(true) {}

    ~TaskExecutor() {
        Stop();
    }

    /**
     * @brief Submit a task for async execution
     * @param task Task to execute
     * @param client_id Client ID for updating task status
     */
    void SubmitTask(const Task& task, const UUID& client_id);

    /**
     * @brief Stop the executor and wait for all tasks to complete
     */
    void Stop();

    /**
     * @brief Check if executor is running
     */
    bool IsRunning() const { return running_.load(); }

private:
    /**
     * @brief Execute a single task
     * @param task Task to execute
     * @param client_id Client ID for updating task status
     */
    void ExecuteTask(const Task& task, const UUID& client_id);

    std::shared_ptr<Client> client_;
    std::shared_ptr<MasterClient> master_client_;
    ThreadPool thread_pool_;
    ReplicaCopyExecutor copy_executor_;
    ReplicaMoveExecutor move_executor_;
    std::atomic<bool> running_;
};

}  // namespace mooncake

