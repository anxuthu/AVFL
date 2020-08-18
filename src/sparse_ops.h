#ifndef _SPARSE_OPS_H_
#define _SPARSE_OPS_H_

#include <armadillo>
#include "fmath.hpp"

/****** exp() in cmath is SLOW! ******/

void Mul(const arma::SpMat<float>& xl, const arma::Col<float>& wl, arma::Col<float>& out) {
	// xl (dlxn), wl (dlx1), out (nx1)
	std::memset(out.begin(), 0, sizeof(float) * out.size());
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		out.at(it.col()) += wl.at(it.row()) * (*it);
	}
}

void LRGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Col<float>& prod, arma::Col<float>& gradl) {
	// xl (dlxn), yl (nx1), prod (nx1), wl (dlx1), gradl (dlx1)
	std::memset(gradl.begin(), 0, gradl.size() * sizeof(float));
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		gradl.at(it.row()) += (prod.at(it.col()) - yl.at(it.col())) * (*it) / xl.n_cols;
	}
}

void LRGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Col<float>& prod, arma::Mat<float>& gradl) {
	// For SAGA. xl (dlxn), yl (nx1), prod (nx1), wl (dlx1), gradl (dlxn)
	std::memset(gradl.begin(), 0, gradl.size() * sizeof(float));
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		gradl.at(it.row(), it.col()) = (prod.at(it.col()) - yl.at(it.col())) * (*it);
	}
}

void LogisticGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Col<float>& prod, arma::Col<float>& gradl) {
	// xl (dlxn), yl (nx1), prod (nx1), wl (dlx1), gradl (dlx1)
	std::memset(gradl.begin(), 0, gradl.size() * sizeof(float));
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		float coeff = yl.at(it.col()) * prod.at(it.col());
		coeff = (1 - 1 / (1 + fmath::exp(-coeff))) * (-yl.at(it.col()));
		gradl.at(it.row()) += coeff * (*it) / xl.n_cols;
	}
}

void LogisticGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Col<float>& prod, arma::Mat<float>& gradl) {
	// for SAGA. xl (dlxn), yl (nx1), prod (nx1), wl (dlx1), gradl (dlxn)
	std::memset(gradl.begin(), 0, gradl.size() * sizeof(float));
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		float coeff = yl.at(it.col()) * prod.at(it.col());
		coeff = (1 - 1 / (1 + fmath::exp(-coeff))) * (-yl.at(it.col()));
		gradl.at(it.row(), it.col()) = coeff * (*it);
	}
}

void LogisticL2Update(const arma::SpMat<float>& xl, const arma::Col<float>& yl, 
		const arma::Col<float>& prod, arma::Col<float>& wl, const float& eta, const float& reg) {
	// for SGD. xl (dlxn), yl (nx1), prod (nx1), wl (dlx1)
	wl *= 1 - eta * reg;
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		float coeff = yl.at(it.col()) * prod.at(it.col());
		coeff = (1 - 1 / (1 + fmath::exp(-coeff))) * (-yl.at(it.col()));
		wl.at(it.row()) -= eta * coeff * (*it) / xl.n_cols;
	}
}

void LogisticL2Update(const arma::Col<float>& full_gradl, const arma::SpMat<float>& xl,
		const arma::Col<float>& yl, const arma::Col<float>& base_prod, const arma::Col<float>& prod,
		arma::Col<float>& wl, const float& eta, const float& reg) {
	// for SVRG. xl (dlxn), yl (nx1), base_prod (nx1), prod (nx1), wl (dlx1)
	wl -= eta * (full_gradl + reg * wl);
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		float coeff1 = yl.at(it.col()) * base_prod.at(it.col());
		coeff1 = (1 - 1 / (1 + fmath::exp(-coeff1))) * (-yl.at(it.col()));
		float coeff2 = yl.at(it.col()) * prod.at(it.col());
		coeff2 = (1 - 1 / (1 + fmath::exp(-coeff2))) * (-yl.at(it.col()));
		wl.at(it.row()) -= eta * (coeff2 - coeff1) * (*it) / xl.n_cols;
	}
}

void LRL2Update(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Col<float>& prod, arma::Col<float>& wl, const float& eta, const float& reg) {
	// for SGD. xl (dlxn), yl (nx1), prod (nx1), wl (dlx1)
	wl *= 1 - eta * reg;
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		wl.at(it.row()) -= eta * (prod.at(it.col()) - yl.at(it.col())) * (*it) / xl.n_cols;
	}
}

void LRL2Update(const arma::Col<float>& full_gradl, const arma::SpMat<float>& xl,
		const arma::Col<float>& yl, const arma::Col<float>& base_prod, const arma::Col<float>& prod,
		arma::Col<float>& wl, const float& eta, const float& reg) {
	// for SVRG. xl (dlxn), yl (nx1), base_prod (nx1), prod (nx1), wl (dlx1)
	wl -= eta * (full_gradl + reg * wl);
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		float coeff1 = base_prod.at(it.col()) - yl.at(it.col());
		float coeff2 = prod.at(it.col()) - yl.at(it.col());
		wl.at(it.row()) -= eta * (coeff2 - coeff1) * (*it) / xl.n_cols;
	}
}

#endif
