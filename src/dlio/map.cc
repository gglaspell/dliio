/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/map.h"
#include "dlio/utils.h"

dlio::MapNode::MapNode(): Node("dlio_map_node") {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&dlio::MapNode::callbackKeyframe, this, std::placeholders::_1), keyframe_sub_opt);

  // FIX: transient_local QoS — late-joining subscribers (e.g. RViz after restart)
  //      receive the last published map immediately without waiting for a new keyframe.
  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "map", rclcpp::QoS(1).transient_local());

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>("save_pcd",
      std::bind(&dlio::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, this->save_pcd_cb_group);

  this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

}

dlio::MapNode::~MapNode() {}

void dlio::MapNode::getParams() {

  this->declare_parameter<std::string>("odom/odom_frame", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);

  this->get_parameter("odom/odom_frame", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);
}

void dlio::MapNode::start() {
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // Convert scan to PCL format
  pcl::PointCloud<PointType>::Ptr keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // Voxel filter
  this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
  this->voxelgrid.setInputCloud(keyframe_pcl);
  this->voxelgrid.filter(*keyframe_pcl);

  // FIX: hold map_mutex while mutating dlio_map to prevent a data race with the
  //      savePCD service callback which runs in a separate callback group thread.
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    *this->dlio_map += *keyframe_pcl;
  }

  // FIX: skip serialisation entirely when nobody is subscribed — pcl::toROSMsg on
  //      a growing cloud is the dominant CPU cost and wasted when count == 0.
  if (this->map_pub->get_subscription_count() == 0) {
    return;
  }

  // FIX: take a snapshot copy outside the lock so the expensive serialisation
  //      does not block the callback (and therefore the keyframe pipeline) longer
  //      than the brief append above.
  pcl::PointCloud<PointType>::Ptr map_snapshot;
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    map_snapshot = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);
  }

  if (map_snapshot->points.size() == map_snapshot->width * map_snapshot->height) {
    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*map_snapshot, map_ros);
    map_ros.header.stamp = this->now();
    map_ros.header.frame_id = this->odom_frame;
    this->map_pub->publish(map_ros);
  }
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  // FIX: snapshot under lock to prevent race with callbackKeyframe
  pcl::PointCloud<PointType>::Ptr m;
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);
  }

  float leaf_size = req->leaf_size;
  std::string p = req->save_path;

  std::cout << std::setprecision(2) << "Saving map to " << p + "/dlio_map.pcd"
    << " with leaf size " << to_string_with_precision(leaf_size, 2) << "... "; std::cout.flush();

  // Voxelize map
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.setInputCloud(m);
  vg.filter(*m);

  // Save map
  int ret = pcl::io::savePCDFileBinary(p + "/dlio_map.pcd", *m);
  res->success = ret == 0;

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}
