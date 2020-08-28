#include <armadillo>
#include "fmath.hpp"

#include "sparse_ops_mc.h"

void Mul(const arma::SpMat<float>& xl, const arma::Mat<float>& wl, arma::Mat<float>& out) {
	// xl (dlxn), wl (cxdl), out (cxn)
	std::memset(out.begin(), 0, sizeof(float) * out.size());
	for (auto it = xl.begin(); it != xl.end(); ++it) {
		int cur_sample = it.col();
		out.col(cur_sample) += wl.col(it.row()) * (*it);
	}
}

void LogisticMCGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Mat<float>& gradl) {
	// xl (dlxn), yl (nx1), prod (cxn), gradl (cxdl); y: 0 ~ c-1
	std::memset(gradl.begin(), 0, sizeof(float) * gradl.size());
	arma::Mat<float> z = arma::exp(prod);
	z.each_col([](arma::Col<float> &col){ col /= arma::accu(col); });

	for (auto it = xl.begin(); it != xl.end(); ++it) {
		int cur_sample = it.col();
		int cur_class = yl.at(it.col());
		int cur_feature = it.row();
		// first term gradient
		gradl.col(cur_feature) += z.col(cur_sample) * (*it) / xl.n_cols;
		// second term gradient
		gradl.at(cur_class, cur_feature) += - (*it) / xl.n_cols;
	}
}

void LogisticMCGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Cube<float>& gradl) {
	// For SAGA. xl (dlxn), yl (nx1), prod (cxn), gradl (cxdlxn); y: 0 ~ c-1
	std::memset(gradl.begin(), 0, sizeof(float) * gradl.size());
	arma::Mat<float> z = arma::exp(prod);
	z.each_col([](arma::Col<float> &col){ col /= arma::accu(col); });

	for (auto it = xl.begin(); it != xl.end(); ++it) {
		int cur_sample = it.col();
		int cur_class = yl.at(it.col());
		int cur_feature = it.row();
		// first term gradient
		gradl.subcube(0, cur_feature, cur_sample, gradl.n_rows-1, cur_feature, cur_sample) += 
			z.col(cur_sample) * (*it);
		// second term gradient
		gradl.at(cur_class, cur_feature, cur_sample) += - (*it);
	}
}

void LogisticMCL2Update(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Mat<float>& wl, const float& eta, const float& reg,
		const float& fraction) {
	// for SGD. xl (dlxn), yl (nx1), prod (cxn), wl (cxdl); y: 0 ~ c-1
	wl *= 1 - eta * reg;
	arma::Mat<float> z = arma::exp(prod);
	z.each_col([](arma::Col<float> &col){ col /= arma::accu(col); });

	for (auto it = xl.begin(); it != xl.end(); ++it) {
		int cur_sample = it.col();
		int cur_class = yl.at(it.col());
		int cur_feature = it.row();
		// first term gradient
		wl.col(cur_feature) -= eta * z.col(cur_sample) * (*it) / xl.n_cols * fraction;
		// second term gradient
		wl.at(cur_class, cur_feature) -= - eta * (*it) / xl.n_cols * fraction;
	}
}

void LogisticMCL2Update(const arma::Mat<float>& full_gradl, const arma::SpMat<float>& xl,
		const arma::Col<float>& yl, const arma::Mat<float>& base_prod, const arma::Mat<float>& prod,
		arma::Mat<float>& wl, const float& eta, const float& reg) {
	// for SVRG. xl (dlxn), yl (nx1), base_prod (cxn), prod (cxn), wl (cxdl); y: 0 ~ c-1
	wl -= eta * (full_gradl + reg * wl);
	arma::Mat<float> z1 = arma::exp(base_prod);
	z1.each_col([](arma::Col<float> &col){ col /= arma::accu(col); });
	arma::Mat<float> z2 = arma::exp(prod);
	z2.each_col([](arma::Col<float> &col){ col /= arma::accu(col); });

	for (auto it = xl.begin(); it != xl.end(); ++it) {
		// first term gradient
		int cur_sample = it.col();
		int cur_feature = it.row();
		wl.col(cur_feature) -= eta * (z2.col(cur_sample) - z1.col(cur_sample))* (*it) / xl.n_cols;
	}
}
