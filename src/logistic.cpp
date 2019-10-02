#include <armadillo>

#include "logistic.h"

using namespace arma;

float logistic(const Mat<float> &data, const Col<float> &target,
		const Col<float> weight) {
	// data: Nxd, target: Nx1, weight: dx1
	Col<float> obj = log(1 + exp(-target % (data * weight)));
	return arma::accu(obj) / obj.size();
}

float logistic(const Col<float> &prod, const Col<float> &target) {
	return accu(log(1 + exp(-target % prod))) / target.size();
}

Col<float> logistic_grad(const Mat<float> &data, const Col<float> &target,
		const Col<float> &weight) {
	// data: Nxd, target: Nx1, weight: dx1
	Col<float> e = arma::exp(-target % (data * weight));
	e = -target % e / (1 + e);
	return data * e / e.size();
}

Col<float> logistic_grad(const Col<float> &prod, const Mat<float> &data,
		const Col<float> &target) {
	// prod: Nx1, data: Nxdl, target: Nx1
	Col<float> e = exp(-target % prod);
	e = -target % e / (1 + e);
	return data.t() * e / e.size(); // dlx1
}
