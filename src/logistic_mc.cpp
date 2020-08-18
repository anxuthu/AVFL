#include <armadillo>

#include "logistic_mc.h"
#include "one_hot.h"

using namespace arma;

// multinomial logistic regression loss (sparse computation)
float logistic_mc(const Mat<float>& prod, const Col<float>& target) {
	// prod (cxn), target (nx1). target value: 0 ~ c-1
	float loss = accu( log( sum( exp(prod), 0))) / prod.n_cols;
	for (int i = 0; i < prod.n_cols; i++) {
		loss -= prod.at(target.at(i), i) / prod.n_cols;
	}

	return loss;
}
