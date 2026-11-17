#include <gtest/gtest.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace {
class ScopedEnvVar {
   public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* original = std::getenv(name)) original_ = original;
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
    }
    ~ScopedEnvVar() {
        if (original_)
            setenv(name_.c_str(), original_->c_str(), 1);
        else
            unsetenv(name_.c_str());
    }
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

   private:
    std::string name_;
    std::optional<std::string> original_;
};

struct TestDirectory {
    std::filesystem::path path =
        std::filesystem::current_path() /
        ("bucket-read-overlap-" + std::to_string(getpid()));
    ~TestDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::mutex gate_mutex;
std::condition_variable gate_cv;
bool gated = false;
bool overlapped = false;
int entered = 0;
int active = 0;
int peak = 0;
int expected_peak = 1;
bool fail_first = false;
}  // namespace

extern "C" ssize_t __real_preadv(int, const struct iovec*, int, off_t);
extern "C" ssize_t __wrap_preadv(int fd, const struct iovec* iov, int count,
                                 off_t offset) {
    bool observed = false;
    bool fail = false;
    bool delay = false;
    {
        std::unique_lock lock(gate_mutex);
        if (gated) {
            observed = true;
            ++entered;
            fail = fail_first && entered == 1;
            delay = expected_peak > 1 && !fail;
            peak = std::max(peak, ++active);
            gate_cv.notify_all();
            if (entered == 1 && expected_peak > 1)
                overlapped =
                    gate_cv.wait_for(lock, std::chrono::seconds(2),
                                     [] { return peak >= expected_peak; });
        }
    }
    // Keep other reads active long enough to observe the worker bound and to
    // prove that an error waits for already-running reads to drain.
    if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto result = fail ? -1 : __real_preadv(fd, iov, count, offset);
    if (fail) errno = EIO;
    if (observed) {
        std::lock_guard lock(gate_mutex);
        --active;
    }
    return result;
}

namespace {
void CheckOverlap(bool single_bucket, bool span_read = false,
                  bool inject_error = false, const char* concurrency = "2",
                  int workers = 2, int key_count = 4) {
    using namespace mooncake;
    TestDirectory directory;
    ScopedEnvVar read_concurrency("MOONCAKE_OFFLOAD_READ_CONCURRENCY",
                                  concurrency);
    ScopedEnvVar use_uring("MOONCAKE_OFFLOAD_USE_URING", "false");
    FileStorageConfig config;
    config.storage_filepath = directory.path.string();
    BucketBackendConfig bucket_config;
    BucketStorageBackend backend(config, bucket_config);
    ASSERT_TRUE(backend.Init());
    std::vector<std::string> values;
    for (int i = 0; i < key_count; ++i)
        values.emplace_back(32, static_cast<char>('A' + i));
    std::unordered_map<std::string, std::vector<Slice>> batch;
    for (int i = 0; i < key_count; ++i) {
        batch.emplace(std::to_string(i),
                      std::vector<Slice>{{values[i].data(), values[i].size()}});
        if (!single_bucket || i == key_count - 1) {
            ASSERT_TRUE(backend.BatchOffload(
                batch, [](const auto&, auto&) { return ErrorCode::OK; }));
            batch.clear();
        }
    }
    std::vector<std::vector<char>> buffers(key_count, std::vector<char>(4096));
    std::unordered_map<std::string, Slice> slices;
    std::unordered_map<std::string, StorageValueSpan> spans;
    for (int i = 0; i < key_count; ++i) {
        slices.emplace(std::to_string(i),
                       Slice{buffers[i].data(), values[i].size()});
        spans.emplace(
            std::to_string(i),
            StorageValueSpan{3, values[i].size(),
                             Slice{buffers[i].data(), values[i].size() - 5}});
    }
    {
        std::lock_guard lock(gate_mutex);
        gated = true;
        overlapped = false;
        entered = active = peak = 0;
        expected_peak = workers;
        fail_first = inject_error;
    }
    auto result =
        span_read ? backend.BatchLoadSpans(spans) : backend.BatchLoad(slices);
    {
        std::lock_guard lock(gate_mutex);
        gated = false;
        if (workers > 1) {
            EXPECT_TRUE(overlapped);
        }
        EXPECT_EQ(active, 0);
        EXPECT_EQ(peak, workers);
        EXPECT_GE(entered, workers);
    }
    if (inject_error) {
        EXPECT_FALSE(result);
    } else {
        ASSERT_TRUE(result);
        for (int i = 0; i < key_count; ++i) {
            const auto expected =
                span_read ? values[i].substr(3, values[i].size() - 5)
                          : values[i];
            EXPECT_EQ(std::string(buffers[i].data(), expected.size()), expected)
                << "key=" << i;
        }
    }
}
}  // namespace

TEST(BucketReadOverlap, PublicBatchLoadOverlapsBuckets) { CheckOverlap(false); }

TEST(BucketReadOverlap, PublicBatchLoadOverlapsKeysInOneBucket) {
    CheckOverlap(true);
}

TEST(BucketReadOverlap, PublicSpanLoadOverlapsUnalignedWindows) {
    CheckOverlap(true, true);
}

TEST(BucketReadOverlap, ReadErrorDrainsOtherWorkers) {
    CheckOverlap(false, false, true);
}

TEST(BucketReadOverlap, DefaultAndInvalidConcurrencyStaySerial) {
    for (const char* setting :
         {static_cast<const char*>(nullptr), "invalid", "0", "-1"}) {
        SCOPED_TRACE(setting ? setting : "unset");
        CheckOverlap(false, false, false, setting, 1);
    }
}

TEST(BucketReadOverlap, ConcurrencyIsCappedAtEight) {
    CheckOverlap(false, false, false, "99", 8, 16);
}
