#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include "nano_gicp/nanoflann_adaptor.h"

namespace nano_gicp {

using Covariance = Eigen::Matrix4f;
using CovarianceList = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
using Mahalanobis = Eigen::Matrix4f;
using MahalanobisList = std::vector<Mahalanobis, Eigen::aligned_allocator<Mahalanobis>>;

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

template<typename PointSource, typename PointTarget>
class NanoGICP : public pcl::Registration<PointSource, PointTarget> {

public:
  using Ptr = std::shared_ptr<NanoGICP<PointSource, PointTarget>>;
  using ConstPtr = std::shared_ptr<const NanoGICP<PointSource, PointTarget>>;
  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

public:
  NanoGICP();
  virtual ~NanoGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);
  void setMaxCorrespondenceDistance(float corr);
  
  void setTransformationEpsilon(float eps);
  void setRotationEpsilon(float eps);
  void setInitialLambdaFactor(float lambda);

  void setPhotometricWeight(float weight);
  void setGradientKNeighbors(int k);

  const CovarianceList& getSourceCovariances() const;
  
  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

  float source_density_ = 0.0f;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) override;
  
  // Signature changed to void return type to match your implementation
  void linearize(const Eigen::Isometry3f& trans, Eigen::Matrix<float, 6, 6>* H, Eigen::Matrix<float, 6, 1>* b);
  void update_correspondences(const Eigen::Isometry3f& trans);

  template<typename PointT>
  void calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, nanoflann::KdTreeFLANN<PointT>& kdtree, CovarianceList& covs);

  bool estimate_spatial_intensity_gradient(int target_index, Eigen::Vector3f& gradient) const;
  void calculate_target_intensity_gradients();

protected:
  using pcl::Registration<PointSource, PointTarget>::reg_name_;
  using pcl::Registration<PointSource, PointTarget>::input_;
  using pcl::Registration<PointSource, PointTarget>::target_;
  using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
  using pcl::Registration<PointSource, PointTarget>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;
  // New members to match your implementation
  float rotation_epsilon_; 
  float lambda_factor_;

  int num_threads_;
  int k_correspondences_;
  RegularizationMethod regularization_method_;

  std::unique_ptr<nanoflann::KdTreeFLANN<PointSource>> input_kdtree_;
  std::unique_ptr<nanoflann::KdTreeFLANN<PointTarget>> target_kdtree_;

  CovarianceList source_covs_;
  CovarianceList target_covs_;

  std::vector<int> correspondences_;
  std::vector<float> sq_distances_;
  MahalanobisList mahalanobis_;

  float photometric_weight_ = 0.0f;
  int gradient_k_neighbors_ = 10;
  float intensity_gradient_threshold_ = 1e-6; // New member from your code
  
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_intensity_gradients_;
  std::vector<bool> gradient_valid_; // New member from your code
};
} // namespace nano_gicp
