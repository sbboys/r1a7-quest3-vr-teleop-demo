#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "unitree/dds_wrapper/robots/g1/g1.h"
#include "unitree/dds_wrapper/robots/r1/r1.h"
#include "unitree/robot/channel/channel_factory.hpp"

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

float arg_float(int argc, char const *argv[], const std::string &name, float value) {
  const std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) {
      return std::stof(arg.substr(prefix.size()));
    }
  }
  return value;
}

std::string arg_string(int argc, char const *argv[], const std::string &name, const std::string &value) {
  const std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) {
      return arg.substr(prefix.size());
    }
  }
  return value;
}

}  // namespace

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <networkInterface> "
              << "[--state_topic=rt/lf/lowstate] [--command_topic=rt/arm_sdk] "
              << "[--duration=8] [--amplitude_deg=3]\n";
    return 2;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  const std::string iface = argv[1];
  const std::string state_topic = arg_string(argc, argv, "state_topic", "rt/lf/lowstate");
  const std::string command_topic = arg_string(argc, argv, "command_topic", "rt/arm_sdk");
  const float duration = arg_float(argc, argv, "duration", 8.0f);
  const float amplitude = arg_float(argc, argv, "amplitude_deg", 3.0f) * float(M_PI) / 180.0f;

  unitree::robot::ChannelFactory::Instance()->Init(0, iface);

  auto lowstate = std::make_shared<unitree::robot::g1::subscription::LowState>(state_topic);
  lowstate->set_timeout_ms(1000);
  std::cout << "[R1-A7 OFFICIAL ARM SDK] waiting for " << state_topic << " ...\n";
  lowstate->wait_for_connection();

  auto armsdk = std::make_unique<unitree::robot::r1::publisher::ArmSdk>(command_topic);
  std::cout << "[R1-A7 OFFICIAL ARM SDK] publisher: " << command_topic << "\n";

  constexpr int kRightShoulderPitch = int(unitree::robot::r1::JointIndex::RightShoulderPitch);
  std::array<float, 35> q0{};
  for (size_t i = 0; i < q0.size() && i < lowstate->msg_.motor_state().size(); ++i) {
    q0[i] = lowstate->msg_.motor_state().at(i).q();
  }

  float q_min = q0[kRightShoulderPitch];
  float q_max = q0[kRightShoulderPitch];

  auto seed_current_pose = [&]() {
    armsdk->lock();
    armsdk->weight(1.0f);
    for (int sdk_i = 0; sdk_i < int(armsdk->JOINTS.size()); ++sdk_i) {
      int idx = int(armsdk->JOINTS[sdk_i]);
      auto &motor = armsdk->msg_.motor_cmd().at(idx);
      motor.mode(1);
      motor.q(lowstate->msg_.motor_state().at(idx).q());
      motor.dq(0.0f);
      motor.tau(0.0f);
      motor.kp(sdk_i == 5 ? 50.0f : 20.0f);
      motor.kd(2.0f);
    }
    armsdk->unlockAndPublish();
  };

  seed_current_pose();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  const auto t0 = std::chrono::steady_clock::now();
  while (!g_stop) {
    const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
    if (t > duration) {
      break;
    }
    const float target = q0[kRightShoulderPitch] + amplitude * std::sin(2.0f * float(M_PI) * 0.25f * t);

    armsdk->lock();
    armsdk->weight(1.0f);
    auto &motor = armsdk->msg_.motor_cmd().at(kRightShoulderPitch);
    motor.mode(1);
    motor.q(target);
    motor.dq(0.0f);
    motor.tau(0.0f);
    motor.kp(50.0f);
    motor.kd(2.0f);
    armsdk->unlockAndPublish();

    const float q = lowstate->msg_.motor_state().at(kRightShoulderPitch).q();
    q_min = std::min(q_min, q);
    q_max = std::max(q_max, q);
    std::cout << "[R1-A7 OFFICIAL ARM SDK] q=" << q << " target=" << target
              << " moved=" << (q_max - q_min) << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  for (int i = 0; i < 80; ++i) {
    const float w = 1.0f - float(i + 1) / 80.0f;
    armsdk->lock();
    armsdk->weight(std::max(0.0f, w));
    armsdk->unlockAndPublish();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::cout << "[R1-A7 OFFICIAL ARM SDK] done moved=" << (q_max - q_min) << "\n";
  if ((q_max - q_min) < 0.01f) {
    std::cerr << "[R1-A7 OFFICIAL ARM SDK] WARNING: command published but lowstate did not move\n";
    return 3;
  }
  return 0;
}
