/*
 * C3 -- ROS 2 node wrapping the optimized C++ fusion core.
 *
 * The node is event-driven end to end:
 *   - the ROS executor delivers a /scan callback,
 *   - the callback hands the encoded frame to the fusion core's worker thread
 *     via a semaphore (the core does not poll),
 *   - the callback waits for the worker and publishes the resulting pose.
 *
 * The core itself is int8 / sparse-CSR / -O3 and is pinned to a configurable
 * set of ARM cores.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fusion_core.h"

namespace {

constexpr int PROFILE_BINS = 256;
constexpr double RMIN = 0.15;
constexpr double RMAX = 12.0;

class FusionCoreNode : public rclcpp::Node {
public:
    FusionCoreNode() : Node("fusion_core_node") {
        weights_path_ = declare_parameter<std::string>("weights_path", "");
        frame_id_ = declare_parameter<std::string>("frame_id", "base_link");
        rate_hz_ = declare_parameter<double>("rate_hz", 50.0);
        cpu_lo_ = declare_parameter<int>("cpu_affinity_lo", 0);
        cpu_hi_ = declare_parameter<int>("cpu_affinity_hi", 0);
        publish_spikes_ = declare_parameter<bool>("publish_spikes", false);

        if (weights_path_.empty()) {
            RCLCPP_ERROR(get_logger(), "weights_path parameter is required");
            throw std::runtime_error("weights_path required");
        }

        fc_ = fc_create(weights_path_.c_str());
        if (!fc_) {
            RCLCPP_ERROR(get_logger(), "fc_create failed for %s", weights_path_.c_str());
            throw std::runtime_error("fc_create failed");
        }
        fc_configure(fc_, nullptr, 0, PROFILE_BINS, static_cast<int>(rate_hz_), cpu_lo_, cpu_hi_);

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom", 20);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("fusion_pose", 20);
        if (publish_spikes_) {
            spike_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("fusion_spikes", 10);
        }
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::LaserScan::SharedPtr msg) { this->on_scan(msg); });

        RCLCPP_INFO(get_logger(),
                    "fusion_core_node ready: int8 sparse CSR core, %d bins, %.1f Hz, cpu[%d..%d]",
                    PROFILE_BINS, rate_hz_, cpu_lo_, cpu_hi_);
    }

    ~FusionCoreNode() override {
        if (fc_) {
            fc_destroy(fc_);
            fc_ = nullptr;
        }
    }

private:
    static void profile(const std::vector<float>& ranges, float* out) {
        const size_t n = ranges.size();
        for (int b = 0; b < PROFILE_BINS; ++b) {
            out[b] = static_cast<float>(RMAX);
        }
        if (n == 0) return;
        for (size_t i = 0; i < n; ++i) {
            const float r = ranges[i];
            if (!std::isfinite(r) || r < RMIN || r > RMAX) continue;
            int b = static_cast<int>((static_cast<double>(i) / static_cast<double>(n)) * PROFILE_BINS);
            if (b < 0) b = 0;
            if (b >= PROFILE_BINS) b = PROFILE_BINS - 1;
            if (r < out[b]) out[b] = r;
        }
        for (int b = 0; b < PROFILE_BINS; ++b) {
            out[b] = static_cast<float>((out[b] - RMIN) / (RMAX - RMIN));
        }
    }

    void on_scan(sensor_msgs::msg::LaserScan::SharedPtr msg) {
        const auto t_cb = std::chrono::steady_clock::now();
        profile(msg->ranges, input_.data());

        const int64_t frame = frame_index_++;
        fc_set_input(fc_, frame, input_.data());
        fc_submit(fc_, frame);
        double lat_ms = 0.0;
        fc_wait(fc_, frame, &lat_ms);

        double x = 0, y = 0, th = 0;
        fc_last_pose(fc_, &x, &y, &th);
        const float* spikes = fc_last_output(fc_);

        const auto t_end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(t_end - t_cb).count();

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = msg->header.stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = x;
        odom.pose.pose.position.y = y;
        odom.pose.pose.orientation.z = std::sin(th * 0.5);
        odom.pose.pose.orientation.w = std::cos(th * 0.5);
        odom_pub_->publish(odom);

        geometry_msgs::msg::PoseStamped ps;
        ps.header = odom.header;
        ps.pose = odom.pose.pose;
        pose_pub_->publish(ps);

        if (spike_pub_) {
            std_msgs::msg::Float32MultiArray arr;
            arr.data.assign(spikes, spikes + 64);
            spike_pub_->publish(arr);
        }

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                             "frame %ld  core %.3f ms  total %.3f ms  pose (%.4f, %.4f, %.4f)",
                             static_cast<long>(frame), lat_ms, total_ms, x, y, th);
    }

    std::string weights_path_;
    std::string frame_id_;
    double rate_hz_ = 50.0;
    int cpu_lo_ = 0, cpu_hi_ = 0;
    bool publish_spikes_ = false;

    Fc* fc_ = nullptr;
    std::vector<float> input_ = std::vector<float>(PROFILE_BINS, 0.0f);
    int64_t frame_index_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr spike_pub_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FusionCoreNode>());
    rclcpp::shutdown();
    return 0;
}
