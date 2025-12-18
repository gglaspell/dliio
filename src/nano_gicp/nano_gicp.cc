#include "nano_gicp/nano_gicp.h"
#include "dlio/dlio.h"
#include <omp.h>
#include <Eigen/Dense>
#include <pcl/common/transforms.h>

template class nano_gicp::NanoGICP<dlio::Point, dlio::Point>;

namespace nano_gicp {

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived>& v) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> m;
    m.setZero();
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 0) = v(2);
    m(1, 2) = -v(0);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::NanoGICP() {
  reg_name_ = "NanoGICP";
  this->num_threads_ = omp_get_max_threads();
  this->k_correspondences_ = 20;
  this->corr_dist_threshold_ = std::numeric_limits<float>::max();
  this->regularization_method_ = RegularizationMethod::PLANE;
  this->max_iterations_ = 64;
  this->transformation_epsilon_ = 1e-4;
  this->rotation_epsilon_ = 1e-4;
  this->lambda_factor_ = 1e-9;
  this->photometric_weight_ = 0.0f;
  this->gradient_k_neighbors_ = 10;
  this->intensity_gradient_threshold_ = 1e-6;
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
void NanoGICP<PointSource, PointTarget>::setMaxCorrespondenceDistance(float corr) {
  this->corr_dist_threshold_ = corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTransformationEpsilon(float eps) {
    this->transformation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRotationEpsilon(float eps) {
    this->rotation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInitialLambdaFactor(float lambda) {
    this->lambda_factor_ = lambda;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setPhotometricWeight(float weight) {
    this->photometric_weight_ = weight;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setGradientKNeighbors(int k) {
    this->gradient_k_neighbors_ = k;
}

template <typename PointSource, typename PointTarget>
const CovarianceList& NanoGICP<PointSource, PointTarget>::getSourceCovariances() const {
    return source_covs_;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  this->regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget>::setInputSource(cloud);
  
  input_kdtree_.reset(new nanoflann::KdTreeFLANN<PointSource>(false));
  input_kdtree_->setInputCloud(cloud);
  
  calculate_covariances(cloud, *input_kdtree_, source_covs_);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);

  target_kdtree_.reset(new nanoflann::KdTreeFLANN<PointTarget>(false));
  target_kdtree_->setInputCloud(cloud);

  calculate_covariances(cloud, *target_kdtree_, target_covs_);
  
  // Only calculate intensity gradients if photometric weight is enabled
  if (photometric_weight_ > 1e-8) {
      calculate_target_intensity_gradients();
  } else {
      target_intensity_gradients_.clear();
      gradient_valid_.clear();
  }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::calculate_target_intensity_gradients() {
    if (!target_) {
        target_intensity_gradients_.clear();
        gradient_valid_.clear();
        return;
    }
    
    target_intensity_gradients_.assign(target_->size(), Eigen::Vector3f::Zero());
    gradient_valid_.assign(target_->size(), false);
    
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for (int i = 0; i < target_->size(); ++i) {
        Eigen::Vector3f gradient;
        if(estimate_spatial_intensity_gradient(i, gradient)) {
            target_intensity_gradients_[i] = gradient;
            gradient_valid_[i] = true;
        }
    }
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::estimate_spatial_intensity_gradient(
  int target_index, Eigen::Vector3f& gradient) const {
    
  if (target_index < 0 || !this->target_kdtree_ || target_index >= this->target_->size()) {
      return false;
  }
  
  const int k_neighbors = this->gradient_k_neighbors_;
  std::vector<int> nn_indices(k_neighbors);
  std::vector<float> nn_dists(k_neighbors);
  
  int found_neighbors = this->target_kdtree_->nearestKSearch(
      this->target_->at(target_index), k_neighbors, nn_indices, nn_dists);
  
  if (found_neighbors < 4) {
    return false;
  }
  
  Eigen::MatrixXf A(found_neighbors, 4);
  Eigen::VectorXf i(found_neighbors);
  
  float mean_intensity = 0.0f;
  for (int j = 0; j < found_neighbors; ++j) {
    const auto& pt = this->target_->at(nn_indices[j]);
    A(j, 0) = pt.x;
    A(j, 1) = pt.y;
    A(j, 2) = pt.z;
    A(j, 3) = 1.0f;
    i(j) = pt.intensity;
    mean_intensity += pt.intensity;
  }
  mean_intensity /= found_neighbors;
  
  // Calculate intensity variance for validation
  float intensity_variance = 0.0f;
  for (int j = 0; j < found_neighbors; ++j) {
      float diff = this->target_->at(nn_indices[j]).intensity - mean_intensity;
      intensity_variance += diff * diff;
  }
  intensity_variance /= found_neighbors;
  
  // Reject if intensity is too uniform
  if (intensity_variance < intensity_gradient_threshold_) {
      return false;
  }
  
  // Solve least squares
  Eigen::Matrix4f AtA = A.transpose() * A;
  Eigen::Vector4f Ati = A.transpose() * i;
  
  // Check condition number
  Eigen::JacobiSVD<Eigen::Matrix4f> svd(AtA);
  float cond = svd.singularValues()(0) / svd.singularValues()(3);
  if (cond > 1e6 || !std::isfinite(cond)) {
      return false;
  }
  
  Eigen::Vector4f g = AtA.ldlt().solve(Ati);
  
  if (g.hasNaN() || !g.allFinite()) {
      return false;
  }
  
  gradient = g.head<3>();
  
  // Reject unreasonably large or small gradients
  float gradient_mag = gradient.norm();
  if (gradient_mag > 100.0f || gradient_mag < intensity_gradient_threshold_) {
      return false;
  }
  
  return true;
}

template<typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(
    PointCloudSource& output, const Eigen::Matrix4f& guess) {
    
    Eigen::Isometry3f trans = Eigen::Isometry3f::Identity();
    trans.matrix() = guess;
    
    for (int i = 0; i < this->max_iterations_; ++i) {
        update_correspondences(trans);
        
        Eigen::Matrix<float, 6, 6> H;
        Eigen::Matrix<float, 6, 1> b;
        
        linearize(trans, &H, &b);
        
        // Add regularization
        float lambda = (lambda_factor_ > 0) ? lambda_factor_ : 1e-6;
        H.diagonal().array() += lambda;
        
        Eigen::Matrix<float, 6, 1> dx = H.ldlt().solve(-b);

        if(dx.hasNaN() || !dx.allFinite()) {
            break;
        }

        // Apply transformation update
        trans.prerotate(Eigen::AngleAxisf(dx[2], Eigen::Vector3f::UnitZ()));
        trans.prerotate(Eigen::AngleAxisf(dx[1], Eigen::Vector3f::UnitY()));
        trans.prerotate(Eigen::AngleAxisf(dx[0], Eigen::Vector3f::UnitX()));
        trans.pretranslate(dx.tail<3>());

        // Check convergence
        if (dx.head<3>().norm() < transformation_epsilon_ && 
            dx.tail<3>().norm() < transformation_epsilon_) {
            break;
        }
    }

    this->final_transformation_ = trans.matrix();
    pcl::transformPointCloud(*this->input_, output, this->final_transformation_);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3f& trans) {
    correspondences_.assign(input_->size(), -1);
    sq_distances_.assign(input_->size(), std::numeric_limits<float>::max());
    mahalanobis_.assign(input_->size(), Eigen::Matrix4f::Identity());

    std::vector<int> k_indices(1);
    std::vector<float> k_sq_dists(1);

    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) \
        firstprivate(k_indices, k_sq_dists)
    for (size_t i = 0; i < input_->size(); ++i) {
        PointTarget transformed_pt;
        transformed_pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

        target_kdtree_->nearestKSearch(transformed_pt, 1, k_indices, k_sq_dists);
        
        float max_dist_sq = this->corr_dist_threshold_ * this->corr_dist_threshold_;
        if (k_sq_dists[0] < max_dist_sq) {
            correspondences_[i] = k_indices[0];
            sq_distances_[i] = k_sq_dists[0];
            
            const Eigen::Matrix4f& source_cov = source_covs_[i];
            const Eigen::Matrix4f& target_cov = target_covs_[k_indices[0]];
            Eigen::Matrix4f RCR = (source_cov + target_cov);
            RCR(3, 3) = 1.0;
            
            mahalanobis_[i] = RCR.inverse();
            mahalanobis_[i](3, 3) = 0.0;
        }
    }
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::linearize(
    const Eigen::Isometry3f& trans, 
    Eigen::Matrix<float, 6, 6>* H, 
    Eigen::Matrix<float, 6, 1>* b) {
    
    H->setZero();
    b->setZero();
    
    std::vector<Eigen::Matrix<float, 6, 6>> H_private(num_threads_, Eigen::Matrix<float, 6, 6>::Zero());
    std::vector<Eigen::Matrix<float, 6, 1>> b_private(num_threads_, Eigen::Matrix<float, 6, 1>::Zero());
    
    bool use_photometric = (photometric_weight_ > 1e-8) && !target_intensity_gradients_.empty();
    
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for(int i = 0; i < input_->size(); ++i) {
        int thread_num = omp_get_thread_num();
        int target_index = correspondences_[i];
        
        if(target_index < 0) {
            continue;
        }
        
        const auto& source_pt = input_->at(i);
        Eigen::Vector4f source_homogeneous = trans * source_pt.getVector4fMap();
        Eigen::Vector3f transformed_source = source_homogeneous.head<3>();
        
        const auto& target_pt = target_->at(target_index);
        Eigen::Vector3f target_pos = target_pt.getVector3fMap();
        
        // Geometric term
        Eigen::Vector3f residual = transformed_source - target_pos;
        Eigen::Matrix<float, 3, 6> J_geometric;
        J_geometric.block<3, 3>(0, 0) = -skew(transformed_source);
        J_geometric.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();
        
        Eigen::Matrix3f M = mahalanobis_[i].block<3, 3>(0, 0);
        
        H_private[thread_num] += J_geometric.transpose() * M * J_geometric;
        b_private[thread_num] += J_geometric.transpose() * M * residual;
        
        // Photometric term
        if (use_photometric && gradient_valid_[target_index]) {
            float intensity_diff = source_pt.intensity - target_pt.intensity;
            Eigen::Vector3f gradient = target_intensity_gradients_[target_index];
            
            if (gradient.norm() > 1e-6 && gradient.norm() < 100.0f) {
                Eigen::Matrix<float, 1, 6> J_photometric;
                J_photometric.block<1, 3>(0, 0) = -gradient.transpose() * skew(transformed_source);
                J_photometric.block<1, 3>(0, 3) = gradient.transpose();
                
                float weight = photometric_weight_;
                H_private[thread_num] += weight * J_photometric.transpose() * J_photometric;
                b_private[thread_num] += weight * J_photometric.transpose() * intensity_diff;
            }
        }
    }
    
    for(int i = 0; i < num_threads_; ++i) {
        (*H) += H_private[i];
        (*b) += b_private[i];
    }
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
void NanoGICP<PointSource, PointTarget>::calculate_covariances(
    const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
    nanoflann::KdTreeFLANN<PointT>& kdtree,
    CovarianceList& covs) {
    
    covs.resize(cloud->size());
    
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for(int i = 0; i < cloud->size(); ++i) {
        std::vector<int> k_indices(k_correspondences_);
        std::vector<float> k_sq_distances(k_correspondences_);
        
        kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
        
        Eigen::Matrix<float, 4, -1> neighbors(4, k_correspondences_);
        for(int j = 0; j < k_indices.size(); ++j) {
            neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap();
        }
        
        neighbors.row(3).array() = 1.0f;
        Eigen::Vector4f mean = neighbors.rowwise().mean();
        Eigen::Matrix<float, 4, -1> centered = neighbors.colwise() - mean;
        centered.row(3).array() = 0.0f;
        
        Eigen::Matrix4f cov = (centered * centered.transpose()) / static_cast<float>(k_correspondences_);
        cov(3, 3) = 1.0;
        
        if(regularization_method_ == RegularizationMethod::PLANE) {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU);
            Eigen::Vector3f values = svd.singularValues();
            values(2) = std::max(values(2), 0.001f);
            cov.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixU().transpose();
        } else {
            cov.block<3, 3>(0, 0) += Eigen::Matrix3f::Identity() * 0.001f;
        }
        
        covs[i] = cov;
    }
}

} // namespace nano_gicp
