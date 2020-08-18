#include <armadillo>

#include "one_hot.h"

arma::Mat<int> one_hot(const arma::Col<float>& label, int n_classes, bool start_from_zero) {
	// label: from 0 to n_classes-1 or 1 to n_classes
	arma::Mat<int> res(label.size(), n_classes, arma::fill::zeros);
	for (int i = 0; i < label.size(); i++) {
		res(i, label(i) - 1 + start_from_zero) = 1;
	}
	return res;
}
