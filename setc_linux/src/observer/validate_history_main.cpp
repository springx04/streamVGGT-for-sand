#include "version_store.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace omnivggt::observer;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--run_dir") {
            std::cerr << "Usage: omnivggt_validate_history --run_dir RUN\n";
            return 2;
        }
        const VersionStore store = VersionStore::open_existing(fs::path(argv[2]));
        const CommitVersion baseline_version = store.latest_snapshot_version();
        CanvasState state = store.load_snapshot_for_version(baseline_version).state;
        for (const DeltaIndexEntry& index : store.delta_index()) {
            if (index.version <= baseline_version) {
                continue;
            }
            const PointCloudDelta delta = store.read_delta(index.version);
            if (delta.from_version != state.version) {
                throw std::runtime_error("delta chain starts at unexpected version " + std::to_string(index.version));
            }
            const CanvasState before = state;
            apply_delta_forward(state, delta);
            CanvasState roundtrip = state;
            apply_delta_backward(roundtrip, delta);
            if (hash_state(roundtrip) != hash_state(before)) {
                throw std::runtime_error("forward/backward roundtrip failed at version " + std::to_string(index.version));
            }
        }
        const CanvasState recovered = store.recover_state();
        // recover_state restores the frame cursor from the frame index after
        // applying deltas.  The raw delta chain has no frame-index metadata,
        // so mirror that final cursor before comparing the two states.
        state.last_frame = store.latest_frame();
        if (hash_state(recovered) != hash_state(state)) {
            throw std::runtime_error("recovered state does not match sequential delta replay");
        }
        if (!store.frame_records().empty()) {
            const ReplayBundle bundle = store.build_replay(store.latest_frame());
            CanvasState replayed = bundle.snapshot.state;
            for (const PointCloudDelta& delta : bundle.deltas) {
                apply_delta_forward(replayed, delta);
            }
            const FrameRecord target = store.frame_at_or_before(store.latest_frame());
            if (replayed.version != target.commit_version) {
                throw std::runtime_error("frame index commit_version does not match replay result");
            }
        }
        std::cout << "history valid: frames=" << store.frame_records().size()
                  << " deltas=" << store.delta_index().size()
                  << " latest_frame=" << store.latest_frame()
                  << " latest_version=" << store.latest_version()
                  << " hash=" << hash_state(state) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
