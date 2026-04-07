#include <gtest/gtest.h>
#include <include/openreg.h>

namespace {

class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(DeviceTest, GetDeviceCountValid) {
  int count = -1;
  EXPECT_EQ(orGetDeviceCount(&count), orSuccess);
  EXPECT_EQ(count, 1);  // Single device configuration
}

TEST_F(DeviceTest, GetDeviceCountNullptr) {
  // orGetDeviceCount should reject null output pointers.
  EXPECT_EQ(orGetDeviceCount(nullptr), orErrorUnknown);
}

TEST_F(DeviceTest, GetDeviceValid) {
  int device = -1;
  EXPECT_EQ(orGetDevice(&device), orSuccess);
  EXPECT_EQ(device, 0);
}

TEST_F(DeviceTest, GetDeviceNullptr) {
  // Defensive path: null output pointer must return an error.
  EXPECT_EQ(orGetDevice(nullptr), orErrorUnknown);
}

TEST_F(DeviceTest, SetDeviceValid) {
  // Only device 0 is available in single-device configuration
  EXPECT_EQ(orSetDevice(0), orSuccess);

  int device = -1;
  EXPECT_EQ(orGetDevice(&device), orSuccess);
  EXPECT_EQ(device, 0);
}

TEST_F(DeviceTest, SetDeviceInvalidNegative) {
  EXPECT_EQ(orSetDevice(-1), orErrorUnknown);
}

TEST_F(DeviceTest, SetDeviceInvalidTooLarge) {
  // Device indices are 0-based and strictly less than DEVICE_COUNT (1).
  EXPECT_EQ(orSetDevice(1), orErrorUnknown);
}

} // namespace
