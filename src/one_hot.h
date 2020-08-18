#ifndef _ONE_HOT_
# define _ONE_HOT_

#include <armadillo>

arma::Mat<int> one_hot(const arma::Col<float>& label, int n_classes, bool start_from_zero=true);
#endif
