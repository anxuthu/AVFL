#ifndef _LOGISTIC_H_
#define _LOGISTIC_H_

#include <armadillo>

float logistic(const arma::Mat<float> &data, const arma::Col<float> &target,
		const arma::Col<float> weight);

float logistic(const arma::Col<float> &prod, const arma::Col<float> &target);

arma::Col<float> logistic_grad(const arma::Mat<float> &data, const arma::Col<float> &target,
		const arma::Col<float> &weight);

arma::Col<float> logistic_grad(const arma::Col<float> &prod, const arma::Mat<float> &data,
		const arma::Col<float> &target);

arma::Col<float> logistic_grad(const arma::Col<float> &prod, const arma::SpMat<float> &data,
		const arma::Col<float> &target);

#endif
