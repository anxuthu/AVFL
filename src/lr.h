#ifndef _LR_H_
#define _LR_H_

#include <armadillo>

float LR(const arma::Col<float>& prod, const arma::Col<float> yl) {
	// prod: nx1, yl: nx1
	return arma::accu(arma::pow(prod - yl, 2)) / 2 / yl.size();
}

#endif
