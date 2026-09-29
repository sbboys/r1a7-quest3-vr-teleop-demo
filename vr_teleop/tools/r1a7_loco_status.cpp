#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include "unitree/robot/r1/loco/r1_loco_client.hpp"

namespace {

struct Args {
  std::string interface = "enx9c69d37d0967";
  bool get = true;
  bool start = false;
  bool stand_up = false;
  float timeout = 5.0f;
};

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--interface IFACE] [--get] [--start] [--stand_up] [--timeout SEC]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value_after = [&](const std::string& key) -> std::string {
      if (arg == key && i + 1 < argc) {
        return std::string(argv[++i]);
      }
      const std::string prefix = key + "=";
      if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
      }
      return "";
    };

    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return false;
    }
    if (arg == "--get") {
      args.get = true;
      continue;
    }
    if (arg == "--start") {
      args.start = true;
      continue;
    }
    if (arg == "--stand_up") {
      args.stand_up = true;
      continue;
    }

    std::string v = value_after("--interface");
    if (!v.empty()) {
      args.interface = v;
      continue;
    }
    v = value_after("--timeout");
    if (!v.empty()) {
      args.timeout = std::stof(v);
      continue;
    }

    std::cerr << "[R1-A7 LOCO] unknown argument: " << arg << "\n";
    usage(argv[0]);
    return false;
  }

  if (args.start && args.stand_up) {
    std::cerr << "[R1-A7 LOCO] choose only one of --start or --stand_up\n";
    return false;
  }
  return true;
}

bool print_state(unitree::robot::r1::LocoClient& client) {
  int fsm_id = -1;
  int fsm_mode = -1;
  int32_t ret_id = client.GetFsmId(fsm_id);
  std::cout << "[R1-A7 LOCO] get_fsm_id ret=" << ret_id;
  if (ret_id == 0) {
    std::cout << " value=" << fsm_id;
  }
  std::cout << "\n";

  int32_t ret_mode = client.GetFsmMode(fsm_mode);
  std::cout << "[R1-A7 LOCO] get_fsm_mode ret=" << ret_mode;
  if (ret_mode == 0) {
    std::cout << " value=" << fsm_mode;
  }
  std::cout << "\n";
  return ret_id == 0 && ret_mode == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return 2;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, args.interface);
  unitree::robot::r1::LocoClient client;
  client.Init();
  client.SetTimeout(args.timeout);

  bool ok = true;
  if (args.get) {
    ok = print_state(client) && ok;
  }

  if (args.start) {
    int32_t ret = client.Start();
    std::cout << "[R1-A7 LOCO] start ret=" << ret << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ok = (ret == 0) && print_state(client) && ok;
  } else if (args.stand_up) {
    int32_t ret = client.StandUp();
    std::cout << "[R1-A7 LOCO] stand_up ret=" << ret << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ok = (ret == 0) && print_state(client) && ok;
  }

  return ok ? 0 : 1;
}
