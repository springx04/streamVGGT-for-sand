#pragma once

#include "version_store.hpp"

#include <cstddef>

namespace omnivggt::observer {

class HistoryCompactor {
public:
    static void compact(VersionStore& store, std::size_t keep_groups) {
        store.compact(keep_groups);
    }
};

}  // namespace omnivggt::observer
