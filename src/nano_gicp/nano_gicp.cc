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
    this->transformation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInitialLambdaFactor(float lambda) {}

template <typename PointSource, typename PointTarget>
const typename nano_gicp::NanoGICP<PointSource, PointTarget>::CovarianceList& NanoGICP<PointSource, PointTarget>::getSourceCovariances() const {
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
  calculate_target_intensity_gradients();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::calculate_target_intensity_gradients() {
    if (!target_) {
        target_intensity_gradients_.clear();
        return;
    }
    target_intensity_gradients_.assign(target_->size(), Eigen::Vector3f::Zero());
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for (int i = 0; i < target_->size(); ++i) {
        Eigen::Vector3f gradient;
        if(estimate_spatial_intensity_gradient(i, gradient)) {
            target_intensity_gradients_[i] = gradient;
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
  int found_neighbors = this->target_kdtree_->nearestKSearch(this->target_->at(target_index), k_neighbors, nn_indices, nn_dists);
  if (found_neighbors < 4) {
    return false;
  }
  Eigen::MatrixXf A(found_neighbors, 4);
  Eigen::VectorXf i(found_neighbors);
  for (int j = 0; j < found_neighbors; ++j) {
    const auto& pt = this->target_->at(nn_indices[j]);
    A(j, 0) = pt.x;
    A(j, 1) = pt.y;
    A(j, 2) = pt.z;
    A(j, 3) = 1.0f;
    i(j) = pt.intensity;
  }
  Eigen::Vector4f g = (A.transpose() * A).ldlt().solve(A.transpose() * i);
  if (g.hasNaN()) {
      return false;
  }
  gradient = g.head<3>();
  return true;
}

template<typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) {
    Eigen::Isometry3f trans = Eigen::Isometry3f::Identity();
    trans.matrix() = guess;

    for (int i = 0; i < this->max_iterations_; ++i) {
        Eigen::Matrix<float, 6, 6> H;
        Eigen::Matrix<float, 6, 1> b;
        linearize(trans, &H, &b);
        Eigen::Matrix<float, 6, 1> dx = H.ldlt().solve(-b);

        if(dx.hasNaN()) {
            PCL_WARN("Solver converged to NaN values. Aborting registration.\n");
            break;
        }

        // --- START OF CRITICAL FIX ---
        // This is the original, correct update logic from dliio.
        // The order of rotation matters immensely.
        trans.prerotate(Eigen::AngleAxisf(dx[2], Eigen::Vector3f::UnitZ()));
        trans.prerotate(Eigen::AngleAxisf(dx[1], Eigen::Vector3f::UnitY()));
        trans.prerotate(Eigen::AngleAxisf(dx[0], Eigen::Vector3f::UnitX()));
        trans.pretranslate(dx.tail<3>());
        // --- END OF CRITICAL FIX ---

        if (dx.norm() < transformation_epsilon_) {
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

    for (size_t i = 0; i < input_->size(); ++i) {
        PointTarget transformed_pt;
        transformed_pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

        target_kdtree_->nearestKSearch(transformed_pt, 1, k_indices, k_sq_dists);
        
        if (k_sq_dists[0] < this->corr_dist_threshold_ * this->corr_dist_threshold_) {
            correspondences_[i] = k_indices[0];
            sq_distances_[i] = k_sq_dists[0];
        }
    }
}

template <typename PointSource, typename PointTarget>
float NanoGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3f& trans, Eigen::Matrix<float, 6, 6>* H, Eigen::Matrix<float, 6, 1>* b) {
  update_correspondences(trans);

  float sum_errors = 0.0f;
  H->setZero();
  b->setZero();

  #pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) continue;

    const Eigen::Vector4f mean_A = input_->at(i).getVector4fMap();
    const Eigen::Vector4f mean_B = target_->at(target_index).getVector4fMap();
    const Eigen::Matrix4f& C1 = source_covs_[i];
    const Eigen::Matrix4f& C2 = target_covs_[target_index];

    const Eigen::Vector4f transed_mean_A = trans * mean_A;
    const Eigen::Vector4f error = mean_B - transed_mean_A;
    const Eigen::Matrix4f C = C2 + trans.matrix() * C1 * trans.matrix().transpose();
    
    Eigen::Matrix4f mahalanobis = C.inverse();
    mahalanobis_[i] = mahalanobis;
    sum_errors += error.transpose() * mahalanobis * error;

    Eigen::Matrix<float, 4, 6> dtdx0 = Eigen::Matrix<float, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skew(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();
    
    Eigen::Matrix<float, 6, 4> jlossexp = -dtdx0.transpose();
    Eigen::Matrix<float, 6, 6> Hi = jlossexp * mahalanobis * jlossexp.transpose();
    Eigen::Matrix<float, 6, 1> bi = jlossexp * mahalanobis * error;

    if (photometric_weight_ > 1e-6f) {
      const Eigen::Vector3f& intensity_gradient = target_intensity_gradients_[target_index];
      if (intensity_gradient.squaredNorm() > 1e-3f) {
        float e_photo = input_->at(i).intensity - target_->at(target_index).intensity;
        Eigen::Matrix<float, 1, 6> J_photo;
        const Eigen::Vector3f transformed_pt = transed_mean_A.head<3>();
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

template<typename PointSource, typename PointTarget>
template<typename PointT>
bool NanoGICP<PointSource, PointTarget>::calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const nanoflann::KdTreeFLANN<PointT>& kdtree, CovarianceList& covariances) {
    covariances.resize(cloud->size());
    #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
    for(int i = 0; i < cloud->size(); ++i) {
        std::vector<int> k_indices;
        std::vector<float> k_sq_dists;
        kdtree.nearestKSearch(cloud->at(i), 20, k_indices, k_sq_dists);

        if (k_indices.size() < 5) {
            covariances[i] = Eigen::Matrix4f::Identity() * 1e-3;
            continue;
        }

        Eigen::Matrix<float, 4, -1> neighbors(4, k_indices.size());
        for(size_t j=0; j<k_indices.size(); j++) {
            neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap();
        }

        Eigen::Vector4f mean = neighbors.rowwise().mean();
        Eigen::Matrix4f cov = (neighbors.colwise() - mean) * (neighbors.colwise() - mean).transpose() / k_indices.size();
        
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f values = Eigen::Vector3f::Ones() * 1e-3;
        values = svd.singularValues().array().max(values.array());
        
        cov.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        
        covariances[i] = cov;
    }
    return true;
}

} // namespace nano_gicp
