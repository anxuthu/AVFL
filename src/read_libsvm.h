#ifndef _READ_LIBSVM_H_
#define _READ_LIBSVM_H_

#include <string>
#include <armadillo>

// return: data (nxd), target(nx1)
std::pair<arma::Mat<float>, arma::Col<float>> ReadLibsvm(std::string filename,
		int start_feature, int end_feature, int num_data);

std::pair<arma::SpMat<float>, arma::Col<float>> ReadLibsvmSp(std::string filename,
		int start_feature, int end_feature, int num_data);

#endif
