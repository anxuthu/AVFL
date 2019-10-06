#ifndef _SPARSE_OPS_H_
#define _SPARSE_OPS_H_

#include <armadillo>
#include <chrono>
#include <vector>
#include "fmath.hpp"

/****** exp() in cmath is SLOW! ******/

void Mul(const arma::SpMat<float>& xl, const arma::Col<float>& wl, arma::Col<float>& out) {
	// xl (dlxn), wl (dlx1), out (nx1)
	std::memset(out.begin(), 0, sizeof(float) * out.size());
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		out.at(it.col()) += wl.at(it.row()) * (*it);
	}
}

void LogisticL2Update(const arma::SpMat<float>& xl, const arma::Col<float>& yl, 
		const arma::Col<float>& prod, arma::Col<float>& wl, const float& eta, const float& reg) {
	// xl (dlxn), yl (nx1), prod (nx1), wl (dlx1)
	std::vector<bool> flag(wl.size(), true);
	float coeff;
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		// lazy regularization
		if (flag.at(it.row())) {
			wl.at(it.row()) *= 1 - eta * reg;
			flag.at(it.row()) = false;
		}

		// logistic grad
		coeff = yl.at(it.col()) * prod.at(it.col());
		coeff = (1 - 1 / (1 + fmath::exp(-coeff))) * (-yl.at(it.col()));
		wl.at(it.row()) -= eta * coeff * (*it);
	}
}

#endif
