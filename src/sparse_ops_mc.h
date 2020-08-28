#ifndef _SPARSE_OPS_MC_
#define _SPARSE_OPS_MC_

#include <armadillo>
void Mul(const arma::SpMat<float>& xl, const arma::Mat<float>& wl, arma::Mat<float>& out);
void LogisticMCGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Mat<float>& gradl);
void LogisticMCGrad(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Cube<float>& gradl);
void LogisticMCL2Update(const arma::SpMat<float>& xl, const arma::Col<float>& yl,
		const arma::Mat<float>& prod, arma::Mat<float>& wl, const float& eta, const float& reg,
		const float& fraction=1);
void LogisticMCL2Update(const arma::Mat<float>& full_gradl, const arma::SpMat<float>& xl,
		const arma::Col<float>& yl, const arma::Mat<float>& base_prod, const arma::Mat<float>& prod,
		arma::Mat<float>& wl, const float& eta, const float& reg);

#endif
