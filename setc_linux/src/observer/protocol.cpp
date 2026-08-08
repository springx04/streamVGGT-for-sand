#include "protocol.hpp"

#include <limits>
#include <stdexcept>

namespace omnivggt::observer {

namespace {

constexpr std::uint32_t kPacketMagic = 0x5047564fU;  // OVGP
constexpr std::uint16_t kPacketSchema = 1U;
constexpr std::size_t kPacketHeaderSize = 12U;

void require_type(const Packet& packet, const MessageType expected) {
    if (packet.type != expected) {
        throw std::runtime_error("unexpected observer packet type");
    }
}

Packet packet_with_data(const MessageType type, const BinaryWriter& writer) {
    return Packet{type, writer.data()};
}

}  // namespace

Packet make_hello(const HelloMessage& message) {
    BinaryWriter writer;
    writer.u32(static_cast<std::uint32_t>(message.width));
    writer.u32(static_cast<std::uint32_t>(message.height));
    writer.u64(message.live_head_frame);
    writer.u64(message.commit_version);
    writer.string(message.run_name);
    return packet_with_data(MessageType::Hello, writer);
}

HelloMessage decode_hello(const Packet& packet) {
    require_type(packet, MessageType::Hello);
    BinaryReader reader(packet.payload);
    HelloMessage message;
    message.width = static_cast<int>(reader.u32());
    message.height = static_cast<int>(reader.u32());
    message.live_head_frame = reader.u64();
    message.commit_version = reader.u64();
    message.run_name = reader.string();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in Hello packet");
    }
    return message;
}

Packet make_snapshot(const Snapshot& snapshot) {
    BinaryWriter writer;
    write_snapshot(writer, snapshot);
    return packet_with_data(MessageType::Snapshot, writer);
}

Snapshot decode_snapshot(const Packet& packet) {
    require_type(packet, MessageType::Snapshot);
    BinaryReader reader(packet.payload);
    Snapshot snapshot = read_snapshot(reader);
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in Snapshot packet");
    }
    return snapshot;
}

Packet make_frame_status(const FrameRecord& record) {
    BinaryWriter writer;
    write_frame_record(writer, record);
    return packet_with_data(MessageType::FrameStatus, writer);
}

FrameRecord decode_frame_status(const Packet& packet) {
    require_type(packet, MessageType::FrameStatus);
    BinaryReader reader(packet.payload);
    FrameRecord record = read_frame_record(reader);
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in FrameStatus packet");
    }
    return record;
}

Packet make_delta(const MessageType type, const PointCloudDelta& delta, const std::uint64_t generation) {
    if (type != MessageType::Delta && type != MessageType::ReplayDelta) {
        throw std::invalid_argument("delta packet must be Delta or ReplayDelta");
    }
    BinaryWriter writer;
    writer.u64(generation);
    write_delta(writer, delta);
    return packet_with_data(type, writer);
}

PointCloudDelta decode_delta(const Packet& packet, std::uint64_t* generation) {
    if (packet.type != MessageType::Delta && packet.type != MessageType::ReplayDelta) {
        throw std::runtime_error("unexpected delta packet type");
    }
    BinaryReader reader(packet.payload);
    const std::uint64_t packet_generation = reader.u64();
    if (generation != nullptr) {
        *generation = packet_generation;
    }
    PointCloudDelta delta = read_delta(reader);
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in Delta packet");
    }
    return delta;
}

Packet make_live_head(const FrameSeq frame_seq, const CommitVersion version) {
    BinaryWriter writer;
    writer.u64(frame_seq);
    writer.u64(version);
    return packet_with_data(MessageType::LiveHead, writer);
}

std::pair<FrameSeq, CommitVersion> decode_live_head(const Packet& packet) {
    require_type(packet, MessageType::LiveHead);
    BinaryReader reader(packet.payload);
    const FrameSeq frame_seq = reader.u64();
    const CommitVersion version = reader.u64();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in LiveHead packet");
    }
    return {frame_seq, version};
}

Packet make_replay_begin(const ReplayBeginMessage& message) {
    BinaryWriter writer;
    writer.u64(message.generation);
    writer.u64(message.target_frame);
    writer.u64(message.target_version);
    write_snapshot(writer, message.snapshot);
    return packet_with_data(MessageType::ReplayBegin, writer);
}

ReplayBeginMessage decode_replay_begin(const Packet& packet) {
    require_type(packet, MessageType::ReplayBegin);
    BinaryReader reader(packet.payload);
    ReplayBeginMessage message;
    message.generation = reader.u64();
    message.target_frame = reader.u64();
    message.target_version = reader.u64();
    message.snapshot = read_snapshot(reader);
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in ReplayBegin packet");
    }
    return message;
}

Packet make_replay_end(const ReplayEndMessage& message) {
    BinaryWriter writer;
    writer.u64(message.generation);
    writer.u64(message.target_frame);
    writer.u64(message.target_version);
    return packet_with_data(MessageType::ReplayEnd, writer);
}

ReplayEndMessage decode_replay_end(const Packet& packet) {
    require_type(packet, MessageType::ReplayEnd);
    BinaryReader reader(packet.payload);
    ReplayEndMessage message;
    message.generation = reader.u64();
    message.target_frame = reader.u64();
    message.target_version = reader.u64();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in ReplayEnd packet");
    }
    return message;
}

Packet make_resync_required(const CommitVersion latest_snapshot_version) {
    BinaryWriter writer;
    writer.u64(latest_snapshot_version);
    return packet_with_data(MessageType::ResyncRequired, writer);
}

CommitVersion decode_resync_required(const Packet& packet) {
    require_type(packet, MessageType::ResyncRequired);
    BinaryReader reader(packet.payload);
    const CommitVersion version = reader.u64();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in ResyncRequired packet");
    }
    return version;
}

Packet make_error(const std::string& message) {
    BinaryWriter writer;
    writer.string(message);
    return packet_with_data(MessageType::Error, writer);
}

std::string decode_error(const Packet& packet) {
    require_type(packet, MessageType::Error);
    BinaryReader reader(packet.payload);
    const std::string message = reader.string();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in Error packet");
    }
    return message;
}

Packet make_subscribe() { return Packet{MessageType::Subscribe, {}}; }

Packet make_replay_request(const ReplayRequestMessage& message) {
    BinaryWriter writer;
    writer.u64(message.generation);
    writer.u64(message.target_frame);
    return packet_with_data(MessageType::ReplayRequest, writer);
}

ReplayRequestMessage decode_replay_request(const Packet& packet) {
    require_type(packet, MessageType::ReplayRequest);
    BinaryReader reader(packet.payload);
    ReplayRequestMessage message;
    message.generation = reader.u64();
    message.target_frame = reader.u64();
    if (reader.remaining() != 0U) {
        throw std::runtime_error("trailing bytes in ReplayRequest packet");
    }
    return message;
}

std::vector<std::uint8_t> encode_packet(const Packet& packet) {
    if (packet.payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("observer packet is too large");
    }
    BinaryWriter writer;
    writer.u32(kPacketMagic);
    writer.u16(kPacketSchema);
    writer.u16(static_cast<std::uint16_t>(packet.type));
    writer.u32(static_cast<std::uint32_t>(packet.payload.size()));
    writer.bytes(packet.payload);
    return writer.data();
}

PacketHeader decode_packet_header(const std::vector<std::uint8_t>& header_bytes) {
    if (header_bytes.size() != kPacketHeaderSize) {
        throw std::invalid_argument("observer packet header must contain 12 bytes");
    }
    BinaryReader reader(header_bytes);
    if (reader.u32() != kPacketMagic || reader.u16() != kPacketSchema) {
        throw std::runtime_error("invalid observer packet header");
    }
    PacketHeader header;
    header.type = static_cast<MessageType>(reader.u16());
    header.payload_size = reader.u32();
    return header;
}

}  // namespace omnivggt::observer
