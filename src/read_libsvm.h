#include <string>
#include <armadillo>

using namespace arma;

// return: data (nxd), target(nx1)
std::pair<Mat<float>, Col<float>> ReadLibsvm(std::string filename,
		int start_feature, int end_feature, int num_data);

std::pair<SpMat<float>, Col<float>> ReadLibsvmSp(std::string filename,
		int start_feature, int end_feature, int num_data);
