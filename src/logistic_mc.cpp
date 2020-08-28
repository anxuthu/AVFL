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

float logistic_mc_acc(const Mat<float>& prod, const Col<float>& target) {
	// prod (cxn), target (nx1). target value: 0 ~ c-1
	int n = prod.n_cols, c = prod.n_rows, correct = 0;
	for (int i = 0; i < n; i++) {
		int argmax = 0;
		for (int j = 1; j < c; j++) {
			if (prod(j,i) > prod(argmax,i)) {
				argmax = j;
			}
		}
		correct += argmax == target(i);
	}
	return correct/(float)n;
}
