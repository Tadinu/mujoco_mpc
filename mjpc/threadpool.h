// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_THREADPOOL_H_
#define MJPC_THREADPOOL_H_

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <utility>

#include <absl/base/attributes.h>

namespace mjpc {

  ABSL_CONST_INIT static thread_local int worker_id_ = -1;
// ThreadPool class
class ThreadPool {
 public:
  // constructor
  explicit ThreadPool(int num_threads): ctr_(0) {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(std::thread(&ThreadPool::WorkerThread, this, i));
    }
  }

  // destructor
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(m_);
      for (int i = 0; i < threads_.size(); i++) {
        queue_.push(nullptr);
      }
      cv_in_.notify_all();
    }
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  int NumThreads() const { return threads_.size(); }

  // returns an ID between 0 and NumThreads() - 1. must be called within
  // worker thread (returns -1 if not).
  static int WorkerId() { return mjpc::worker_id_; }

  // ----- methods ----- //
  // set task for threadpool
  inline void Schedule(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(m_);
    queue_.push(std::move(task));
    cv_in_.notify_one();
  }

  // return number of tasks completed
  std::uint64_t GetCount() { return ctr_; }

  // reset count to zero
  void ResetCount() { ctr_ = 0; }

  // wait for count, then return
  void WaitCount(int value) {
    std::unique_lock<std::mutex> lock(m_);
    cv_ext_.wait(lock, [&]() { return this->GetCount() >= value; });
  }

 private:
  // ----- methods ----- //

  // execute task with available thread
  
  // ThreadPool worker
  inline void WorkerThread(int i) {
    worker_id_ = i;
    while (true) {
      auto task = [&]() {
        std::unique_lock<std::mutex> lock(m_);
        cv_in_.wait(lock, [&]() { return !queue_.empty(); });
        std::function<void()> task = std::move(queue_.front());
        queue_.pop();
        cv_in_.notify_one();
        return task;
      }();
      if (task == nullptr) {
        {
          std::unique_lock<std::mutex> lock(m_);
          ++ctr_;
          cv_ext_.notify_one();
        }
        break;
      }
      task();

      {
        std::unique_lock<std::mutex> lock(m_);
        ++ctr_;
        cv_ext_.notify_one();
      }
    }
  }
  
  // ----- members ----- //
  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_in_;
  std::condition_variable cv_ext_;
  std::queue<std::function<void()>> queue_;
  std::uint64_t ctr_;
};

}  // namespace mjpc

#endif  // MJPC_THREADPOOL_H_
