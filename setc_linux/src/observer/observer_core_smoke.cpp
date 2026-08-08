#include "version_store.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace omnivggt::observer;
namespace fs = std::filesystem;

int main() {
    try {
        CanvasState initial;
        initial.reset(8, 6);
        const std::uint64_t before_hash = hash_state(initial);
        CandidatePatch first;
        first.frame_seq = 0;
        first.base_version = 0;
        first.width = initial.width;
        first.height = initial.height;
        first.initialize_canvas = true;
        first.anchor_rgba.assign(initial.slot_count(), pack_rgba(20, 30, 40));
        first.anchor_camera = AnchorCamera{100.0f, 100.0f, 4.0f, 3.0f, 1.0f, 0.0f};
        first.updates.push_back(SlotUpdate{10, SlotValue{2.0f, 0.9f, pack_rgba(255, 0, 0), 0, 1}});
        first.observed_slots.push_back(10);
        const PointCloudDelta delta = commit_patch(initial, first);
        if (delta.to_version != 1U || initial.valid[10] == 0U) {
            throw std::runtime_error("initial commit failed");
        }
        CanvasState restored = initial;
        apply_delta_backward(restored, delta);
        if (hash_state(restored) != before_hash) {
            throw std::runtime_error("delta backward roundtrip failed");
        }
        apply_delta_forward(restored, delta);
        if (hash_state(restored) != hash_state(initial)) {
            throw std::runtime_error("delta forward roundtrip failed");
        }

        CandidatePatch second;
        second.frame_seq = 1;
        second.base_version = initial.version;
        second.width = initial.width;
        second.height = initial.height;
        second.observed_slots.push_back(20);
        second.updates.push_back(SlotUpdate{20, SlotValue{3.0f, 0.8f, pack_rgba(0, 255, 0), 1, 1}});
        const PointCloudDelta second_delta = commit_patch(initial, second);
        if (second_delta.from_version != 1U || second_delta.to_version != 2U || initial.valid[20] == 0U) {
            throw std::runtime_error("second commit failed");
        }

        const fs::path temp = fs::temp_directory_path() / "omnivggt_observer_core_smoke";
        std::error_code error;
        fs::remove_all(temp, error);
        CanvasState store_initial;
        store_initial.reset(initial.width, initial.height);
        VersionStore store = VersionStore::create_new(temp, store_initial, 1);
        const std::uint64_t delta_offset = store.append_delta(delta);
        FrameRecord record;
        record.frame_seq = delta.frame_seq;
        record.base_version = delta.from_version;
        record.commit_version = delta.to_version;
        record.status = FrameStatus::Committed;
        record.changed_point_count = static_cast<std::uint32_t>(delta.changes.size());
        record.valid_point_count = delta.valid_point_count;
        record.image_name = "smoke.png";
        store.append_frame(record, delta_offset);
        const std::uint64_t second_delta_offset = store.append_delta(second_delta);
        FrameRecord second_record = record;
        second_record.frame_seq = second_delta.frame_seq;
        second_record.base_version = second_delta.from_version;
        second_record.commit_version = second_delta.to_version;
        second_record.image_name = "smoke_2.png";
        store.append_frame(second_record, second_delta_offset);
        const VersionStore reopened = VersionStore::open_existing(temp);
        const CanvasState recovered = reopened.recover_state();
        if (hash_state(recovered) != hash_state(initial)) {
            throw std::runtime_error("snapshot plus delta recovery failed");
        }
        const ReplayBundle replay = reopened.build_replay(1);
        CanvasState replayed = replay.snapshot.state;
        for (const PointCloudDelta& replay_delta : replay.deltas) {
            apply_delta_forward(replayed, replay_delta);
        }
        if (hash_state(replayed) != hash_state(initial)) {
            throw std::runtime_error("replay bundle recovery failed");
        }
        fs::remove_all(temp, error);
        std::cout << "observer core smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
