#pragma once

#include <atomic>
#include <memory>

#include "rpc_types.h"

namespace mooncake {

class Client;
tl::expected<void, ErrorCode> WriteColdLocalSlices(
    const std::shared_ptr<Client>& client, int fd, uint64_t offset,
    const std::vector<Slice>& slices, int64_t deadline_ms);

inline constexpr uint64_t kColdWriteChunkBytes = 4 * 1024 * 1024;
inline constexpr size_t kColdWriteSlots = 8;

struct ColdWriteBuffer {
    int fd{-1};
    uint64_t size{0};
    std::shared_ptr<std::atomic<uint64_t>> reserved_bytes;
    ~ColdWriteBuffer();
};

struct ColdWriteRequest {
    UUID writer;
    UUID owner;
    ReplicaID reservation{0};
    std::string tenant;
    std::string key;
    uint64_t size{0};
    int64_t deadline_ms{0};
    std::optional<uint64_t> checksum;
    bool ranged{false};
};
YLT_REFL(ColdWriteRequest, writer, owner, reservation, tenant, key, size,
         deadline_ms, checksum, ranged);

struct ColdWriteChunk {
    ColdWriteRequest request;
    std::string source_endpoint;
    uint64_t source_address{0};
    uint64_t offset{0};
    uint64_t size{0};
};
YLT_REFL(ColdWriteChunk, request, source_endpoint, source_address, offset,
         size);

}  // namespace mooncake
