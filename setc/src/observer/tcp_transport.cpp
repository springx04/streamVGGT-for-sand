#include "tcp_transport.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace omnivggt::observer {

namespace {

std::string socket_error(const char* operation) {
#ifdef _WIN32
    return std::string(operation) + " failed with WSA error " + std::to_string(WSAGetLastError());
#else
    return std::string(operation) + " failed: " + std::strerror(errno);
#endif
}

void close_handle(const TcpSocket::Handle handle) noexcept {
    if (handle == TcpSocket::kInvalid) {
        return;
    }
#ifdef _WIN32
    ::closesocket(handle);
#else
    ::close(handle);
#endif
}

}  // namespace

SocketRuntime::SocketRuntime() {
#ifdef _WIN32
    WSADATA data{};
    if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        throw std::runtime_error(socket_error("WSAStartup"));
    }
#endif
}

SocketRuntime::~SocketRuntime() {
#ifdef _WIN32
    ::WSACleanup();
#endif
}

TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket&& other) noexcept : handle_(other.handle_) {
    other.handle_ = kInvalid;
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept {
    if (this != &other) {
        close();
        handle_ = other.handle_;
        other.handle_ = kInvalid;
    }
    return *this;
}

TcpSocket TcpSocket::connect_to(const std::string& host, const std::uint16_t port) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    const std::string port_text = std::to_string(port);
    struct addrinfo* results = nullptr;
    const int result = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &results);
    if (result != 0) {
#ifdef _WIN32
        throw std::runtime_error("getaddrinfo failed: " + std::to_string(result));
#else
        throw std::runtime_error(std::string("getaddrinfo failed: ") + gai_strerror(result));
#endif
    }

    TcpSocket socket;
    for (struct addrinfo* entry = results; entry != nullptr; entry = entry->ai_next) {
        const Handle handle = ::socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
        if (handle == kInvalid) {
            continue;
        }
        if (::connect(handle, entry->ai_addr, static_cast<int>(entry->ai_addrlen)) == 0) {
            socket.handle_ = handle;
            break;
        }
        close_handle(handle);
    }
    ::freeaddrinfo(results);
    if (!socket.valid()) {
        throw std::runtime_error(socket_error("connect"));
    }
    return socket;
}

void TcpSocket::close() noexcept {
    if (valid()) {
        close_handle(handle_);
        handle_ = kInvalid;
    }
}

void TcpSocket::shutdown_both() noexcept {
    if (!valid()) {
        return;
    }
#ifdef _WIN32
    ::shutdown(handle_, SD_BOTH);
#else
    ::shutdown(handle_, SHUT_RDWR);
#endif
}

void TcpSocket::send_all(const std::uint8_t* data, const std::size_t size) const {
    std::size_t offset = 0;
    while (offset < size) {
        const auto remaining = size - offset;
#ifdef _WIN32
        const int sent = ::send(handle_, reinterpret_cast<const char*>(data + offset), static_cast<int>(remaining), 0);
#else
        const ssize_t sent = ::send(handle_, data + offset, remaining, MSG_NOSIGNAL);
#endif
        if (sent <= 0) {
            throw std::runtime_error(socket_error("send"));
        }
        offset += static_cast<std::size_t>(sent);
    }
}

bool TcpSocket::receive_all(std::uint8_t* data, const std::size_t size) const {
    std::size_t offset = 0;
    while (offset < size) {
#ifdef _WIN32
        const int received = ::recv(handle_, reinterpret_cast<char*>(data + offset), static_cast<int>(size - offset), 0);
#else
        const ssize_t received = ::recv(handle_, data + offset, size - offset, 0);
#endif
        if (received == 0) {
            return false;
        }
        if (received < 0) {
            throw std::runtime_error(socket_error("recv"));
        }
        offset += static_cast<std::size_t>(received);
    }
    return true;
}

void TcpSocket::send_packet(const Packet& packet) const {
    const std::vector<std::uint8_t> bytes = encode_packet(packet);
    send_all(bytes.data(), bytes.size());
}

bool TcpSocket::receive_packet(Packet& packet, const std::uint32_t max_payload_size) const {
    constexpr std::size_t kHeaderSize = 12U;
    std::vector<std::uint8_t> header(kHeaderSize);
    if (!receive_all(header.data(), header.size())) {
        return false;
    }
    const PacketHeader decoded = decode_packet_header(header);
    if (decoded.payload_size > max_payload_size) {
        throw std::runtime_error("observer packet exceeds configured maximum size");
    }
    packet.type = decoded.type;
    packet.payload.resize(decoded.payload_size);
    if (!packet.payload.empty() && !receive_all(packet.payload.data(), packet.payload.size())) {
        throw std::runtime_error("connection closed in the middle of observer packet");
    }
    return true;
}

TcpListener::~TcpListener() { close(); }

void TcpListener::listen_on(const std::uint16_t port, const int backlog) {
    close();
    handle_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (handle_ == TcpSocket::kInvalid) {
        throw std::runtime_error(socket_error("socket"));
    }
    int reuse = 1;
    ::setsockopt(handle_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (::bind(handle_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close();
        throw std::runtime_error(socket_error("bind"));
    }
    if (::listen(handle_, backlog) != 0) {
        close();
        throw std::runtime_error(socket_error("listen"));
    }
}

TcpSocket TcpListener::accept_one() const {
    if (!valid()) {
        throw std::runtime_error("listener is not open");
    }
    sockaddr_storage address{};
#ifdef _WIN32
    int address_size = sizeof(address);
#else
    socklen_t address_size = sizeof(address);
#endif
    const TcpSocket::Handle client = ::accept(
        handle_, reinterpret_cast<sockaddr*>(&address), &address_size);
    if (client == TcpSocket::kInvalid) {
        throw std::runtime_error(socket_error("accept"));
    }
    return TcpSocket(client);
}

void TcpListener::close() noexcept {
    if (valid()) {
        // Wake a thread that may currently be blocked in accept().  On Linux,
        // close() from another thread alone is not a reliable way to interrupt
        // a blocking accept(), while shutdown() makes the listener unusable and
        // causes the blocked call to return.
#ifdef _WIN32
        ::shutdown(handle_, SD_BOTH);
#else
        ::shutdown(handle_, SHUT_RDWR);
#endif
        close_handle(handle_);
        handle_ = TcpSocket::kInvalid;
    }
}

}  // namespace omnivggt::observer
