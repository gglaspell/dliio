#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include "nano_gicp/lsq_registration.h"
#include "nano_gicp/nanoflann_adaptor.h"

namespace nano_gicp {

template <typename PointT>
using PointCloudPtr = typename pcl::PointCloud<PointT>::Ptr;
template <typename PointT>
using PointCloudConstPtr = typename pcl::PointCloud<PointT>::ConstPtr;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

template<typename PointSource, typename PointTarget>
class NanoGICP : public LsqRegistration<PointSource, PointTarget> {

public:
  using Scalar = typename pcl::Registration<PointSource, PointTarget, double>::Scalar;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, double>::Matrix4;
  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, double>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, double>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

protected:
  using LsqRegistration<PointSource, PointTarget>::reg_name_;
  using LsqRegistration<PointSource, PointTarget>::input_;
  using LsqRegistration<PointSource, PointTarget>::target_;
  using LsqRegistration<PointSource, PointTarget>::corr_dist_threshold_;
  using LsqRegistration<PointSource, PointTarget>::source_covs_;
  using LsqRegistration<PointSource, PointTarget>::target_covs_;

public:
  NanoGICP();
  virtual ~NanoGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setMaxCorrespondenceDistance(double corr);
  void setRegularizationMethod(RegularizationMethod method);

  void setPhotometricWeight(double weight) {
    photometric_weight_ = weight;
  }

  void setGradientKNeighbors(int k) {
    gradient_k_neighbors_ = k;
  }

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  void update_correspondences(const Eigen::Isometry3d& trans);

  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const nanoflann::KdTreeFLANN<PointT>& kdtree, CovarianceList& covariances);

  bool estimate_spatial_intensity_gradient(int target_index, Eigen::Vector3d& gradient) const;
  
  void calculate_target_intensity_gradients();

protected:
  int num_threads_;
  int k_correspondences_;
  RegularizationMethod regularization_method_;

  std::unique_ptr<nanoflann::KdTreeFLANN<PointSource>> input_kdtree_;
  std::unique_ptr<nanoflann::KdTreeFLANN<PointTarget>> target_kdtree_;
  std::vector<int> correspondences_;
  std::vector<double> sq_distances_;
  MahalanobisList mahalanobis_;

  double photometric_weight_ = 0.0;
  int gradient_k_neighbors_ = 10;
  
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> target_intensity_gradients_;
};

} // namespace nano_gicp
