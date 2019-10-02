#include <armadillo>

using namespace arma;

float logistic(const Mat<float> &data, const Col<float> &target,
		const Col<float> weight);

float logistic(const Col<float> &prod, const Col<float> &target);

Col<float> logistic_grad(const Mat<float> &data, const Col<float> &target,
		const Col<float> &weight);

Col<float> logistic_grad(const Col<float> &prod, const Mat<float> &data,
		const Col<float> &target);
