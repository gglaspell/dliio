#include "nano_gicp/nano_gicp.h"
#include "dlio/dlio.h" // <-- FIX 1: Add missing header
#include <omp.h>
#include <Eigen/Dense>

// FIX 2: Correct template instantiation
template class nano_gicp::NanoGICP<dlio::Point, dlio::Point>;

namespace nano_gicp {

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::NanoGICP() {
  reg_name_ = "NanoGICP";
  this->num_threads_ = omp_get_max_threads();
  this->k_correspondences_ = 20;
  this->corr_dist_threshold_ = std::numeric_limits<double>::max();
  this->regularization_method_ = RegularizationMethod::PLANE;
  this->photometric_weight_ = 0.0;
  this->gradient_k_neighbors_ = 10;
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::~NanoGICP() {}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setNumThreads(int n) {
  this->num_threads_ = n;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  this->k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setMaxCorrespondenceDistance(double corr) {
  this->corr_dist_threshold_ = corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  this->regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, double>::setInputSource(cloud);
  // FIX 3: Initialize input_kdtree_ and dereference source_covs_
  input_kdtree_.reset(new nanoflann::KdTreeFLANN<PointSource>(*cloud));
  calculate_covariances(cloud, *input_kdtree_, *source_covs_);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, double>::setInputTarget(cloud);
  target_kdtree_.reset(new nanoflann::KdTreeFLANN<PointTarget>(*cloud));
  calculate_covariances(cloud, *target_kdtree_, *target_covs_);
  calculate_target_intensity_gradients();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::calculate_target_intensity_gradients() {
    if (!target_) {
        target_intensity_gradients_.clear();
        return;
    }
    target_intensity_gradients_.assign(target_->size(), Eigen::Vector3d::Zero());
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for (int i = 0; i < target_->size(); ++i) {
        Eigen::Vector3d gradient;
        if(estimate_spatial_intensity_gradient(i, gradient)) {
            target_intensity_gradients_[i] = gradient;
        }
    }
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::estimate_spatial_intensity_gradient(
  int target_index,
  Eigen::Vector3d& gradient) const {
  if (target_index < 0 || !this->target_kdtree_ || target_index >= this->target_->size()) {
      return false;
  }
  const int k_neighbors = this->gradient_k_neighbors_;
  std::vector<int> nn_indices(k_neighbors);
  std::vector<float> nn_dists(k_neighbors);
  int found_neighbors = this->target_kdtree_->nearestKSearch(this->target_->at(target_index), k_neighbors, nn_indices, nn_dists);
  if (found_neighbors < 4) {
    return false;
  }
  Eigen::MatrixXd A(found_neighbors, 4);
  Eigen::VectorXd i(found_neighbors);
  for (int j = 0; j < found_neighbors; ++j) {
    const auto& pt = this->target_->at(nn_indices[j]);
    A(j, 0) = pt.x;
    A(j, 1) = pt.y;
    A(j, 2) = pt.z;
    A(j, 3) = 1.0;
    i(j) = pt.intensity;
  }
  Eigen::Vector4d g = (A.transpose() * A).ldlt().solve(A.transpose() * i);
  if (g.hasNaN()) {
      return false;
  }
  gradient = g.head<3>();
  return true;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);
  double sum_errors = 0.0;
  if (H && b) {
    H->setZero();
    b->setZero();
  }
#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }
    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    
    // FIX 4: Dereference shared_ptr before indexing
    const Eigen::Matrix4d& C1 = (*source_covs_)[i];
    const Eigen::Matrix4d& C2 = (*target_covs_)[target_index];
    
    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;
    const Eigen::Matrix4d C = C2 + trans.matrix() * C1 * trans.matrix().transpose();
    Eigen::Matrix4d mahalanobis = C.inverse();

    // FIX 5: Use brackets [] for indexing, not parentheses ()
    mahalanobis_[i] = mahalanobis;
    
    sum_errors += error.transpose() * mahalanobis * error;
    if (H == nullptr || b == nullptr) {
      continue;
    }
    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 4> jlossexp = -dtdx0.transpose();
    Eigen::Matrix<double, 6, 6> Hi = jlossexp * mahalanobis * jlossexp.transpose();
    Eigen::Matrix<double, 6, 1> bi = jlossexp * mahalanobis * error;
    if (photometric_weight_ > 1e-6) {
      const Eigen::Vector3d& intensity_gradient = target_intensity_gradients_[target_index];
      if (intensity_gradient.squaredNorm() > 1e-3) {
        const auto& source_point = this->input_->at(i);
        const auto& target_point = this->target_->at(target_index);
        double e_photo = source_point.intensity - target_point.intensity;
        Eigen::Matrix<double, 1, 6> J_photo;
        const Eigen::Vector3d transformed_pt = transed_mean_A.head<3>();
        J_photo(0, 0) = intensity_gradient.x();
        J_photo(0, 1) = intensity_gradient.y();
        J_photo(0, 2) = intensity_gradient.z();
        J_photo(0, 3) = transformed_pt.y() * intensity_gradient.z() - transformed_pt.z() * intensity_gradient.y();
        J_photo(0, 4) = transformed_pt.z() * intensity_gradient.x() - transformed_pt.x() * intensity_gradient.z();
        J_photo(0, 5) = transformed_pt.x() * intensity_gradient.y() - transformed_pt.y() * intensity_gradient.x();
        Hi += J_photo.transpose() * photometric_weight_ * J_photo;
        bi -= J_photo.transpose() * photometric_weight_ * e_photo;
      }
    }
    #pragma omp critical
    {
      (*H) += Hi;
      (*b) += bi;
    }
  }
  return sum_errors;
}

// ... (Rest of the file remains the same)

} // namespace nano_gicp
