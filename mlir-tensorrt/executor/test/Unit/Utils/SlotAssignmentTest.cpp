#include "mlir-executor/Target/SlotAssignment/SlotAssignment.h"

#include "gtest/gtest.h"

using namespace mlir;

TEST(ParallelSwapTest, Simple) {
  SmallVector<int32_t> sourceSlots = {0, 1, 2, 3};
  SmallVector<int32_t> targetSlots = {0, 1, 2, 3};

  std::string result;
  emitSlotSwap(sourceSlots, targetSlots, 99,
               [&result](int32_t sourceSlot, int32_t targetSlot) {
                 if (!result.empty()) {
                   result += ",";
                 }
                 result += std::to_string(sourceSlot) + "->" +
                           std::to_string(targetSlot);
               });

  EXPECT_EQ(result, "0->0,1->1,2->2,3->3");
}

TEST(ParallelSwapTest, WithCycle) {
  SmallVector<int32_t> sourceSlots = {0, 1, 2, 3};
  SmallVector<int32_t> targetSlots = {3, 2, 1, 0};

  std::string result;
  emitSlotSwap(sourceSlots, targetSlots, 99,
               [&result](int32_t sourceSlot, int32_t targetSlot) {
                 if (!result.empty()) {
                   result += ",";
                 }
                 result += std::to_string(sourceSlot) + "->" +
                           std::to_string(targetSlot);
               });

  EXPECT_EQ(result, "0->99,3->0,99->3,1->99,2->1,99->2");
}
