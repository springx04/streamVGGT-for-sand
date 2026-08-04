#pragma once

#include "stream_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivggt::observer {

enum class MessageType : std::uint16_t {
    Hello = 1,
    Snapshot = 2,
    FrameStatus = 3,
    Delta = 4,
    LiveHead = 5,
    ReplayBegin = 6,
    ReplayDelta = 7,
    ReplayEnd = 8,
    ResyncRequired = 9,
    Error = 10,

    Subscribe = 100,
    ReplayRequest = 101,
};

struct Packet {
    MessageType type = MessageType::Error;
    std::vector<std::uint8_t> payload;
};

struct PacketHeader {
    MessageType type = MessageType::Error;
    std::uint32_t payload_size = 0;
};

struct HelloMessage {
    int width = 0;
    int height = 0;
    FrameSeq live_head_frame = 0;
    CommitVersion commit_version = 0;
    std::string run_name;
};

struct ReplayBeginMessage {
    std::uint64_t generation = 0;
    FrameSeq target_frame = 0;
    CommitVersion target_version = 0;
    Snapshot snapshot;
};

struct ReplayEndMessage {
    std::uint64_t generation = 0;
    FrameSeq target_frame = 0;
    CommitVersion target_version = 0;
};

struct ReplayRequestMessage {
    std::uint64_t generation = 0;
    FrameSeq target_frame = 0;
};

Packet make_hello(const HelloMessage& message);
HelloMessage decode_hello(const Packet& packet);
Packet make_snapshot(const Snapshot& snapshot);
Snapshot decode_snapshot(const Packet& packet);
Packet make_frame_status(const FrameRecord& record);
FrameRecord decode_frame_status(const Packet& packet);
Packet make_delta(MessageType type, const PointCloudDelta& delta, std::uint64_t generation = 0);
PointCloudDelta decode_delta(const Packet& packet, std::uint64_t* generation = nullptr);
Packet make_live_head(FrameSeq frame_seq, CommitVersion version);
std::pair<FrameSeq, CommitVersion> decode_live_head(const Packet& packet);
Packet make_replay_begin(const ReplayBeginMessage& message);
ReplayBeginMessage decode_replay_begin(const Packet& packet);
Packet make_replay_end(const ReplayEndMessage& message);
ReplayEndMessage decode_replay_end(const Packet& packet);
Packet make_resync_required(CommitVersion latest_snapshot_version);
CommitVersion decode_resync_required(const Packet& packet);
Packet make_error(const std::string& message);
std::string decode_error(const Packet& packet);
Packet make_subscribe();
Packet make_replay_request(const ReplayRequestMessage& message);
ReplayRequestMessage decode_replay_request(const Packet& packet);

std::vector<std::uint8_t> encode_packet(const Packet& packet);
PacketHeader decode_packet_header(const std::vector<std::uint8_t>& header_bytes);

}  // namespace omnivggt::observer
