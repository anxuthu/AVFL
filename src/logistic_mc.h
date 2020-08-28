#ifndef _LOGISTIC_MC_H_
#define _LOGISTIC_MC_H_

#include <armadillo>

// multinomial logistic regression
float logistic_mc(const arma::Mat<float>& prod, const arma::Col<float>& target);
float logistic_mc_acc(const arma::Mat<float>& prod, const arma::Col<float>& target);

#endif
