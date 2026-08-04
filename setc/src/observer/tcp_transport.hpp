#pragma once

#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#else
using NativeSocket = int;
#endif

namespace omnivggt::observer {

class SocketRuntime {
public:
    SocketRuntime();
    ~SocketRuntime();
    SocketRuntime(const SocketRuntime&) = delete;
    SocketRuntime& operator=(const SocketRuntime&) = delete;
};

class TcpSocket {
public:
#ifdef _WIN32
    using Handle = SOCKET;
    static constexpr Handle kInvalid = INVALID_SOCKET;
#else
    using Handle = int;
    static constexpr Handle kInvalid = -1;
#endif

    TcpSocket() = default;
    explicit TcpSocket(Handle handle) : handle_(handle) {}
    ~TcpSocket();
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;

    static TcpSocket connect_to(const std::string& host, std::uint16_t port);
    bool valid() const noexcept { return handle_ != kInvalid; }
    Handle handle() const noexcept { return handle_; }
    void close() noexcept;
    void shutdown_both() noexcept;
    void send_packet(const Packet& packet) const;
    bool receive_packet(Packet& packet, std::uint32_t max_payload_size = 256U * 1024U * 1024U) const;

private:
    void send_all(const std::uint8_t* data, std::size_t size) const;
    bool receive_all(std::uint8_t* data, std::size_t size) const;
    Handle handle_ = kInvalid;
};

class TcpListener {
public:
    TcpListener() = default;
    ~TcpListener();
    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;

    void listen_on(std::uint16_t port, int backlog = 16);
    TcpSocket accept_one() const;
    bool valid() const noexcept { return handle_ != TcpSocket::kInvalid; }
    void close() noexcept;

private:
    TcpSocket::Handle handle_ = TcpSocket::kInvalid;
};

}  // namespace omnivggt::observer
