#include <fstream>
#include <string>
#include <armadillo>

#include "read_libsvm.h"

// return: data (nxd), target(nx1)
std::pair<arma::Mat<float>, arma::Col<float>> ReadLibsvm(std::string filename,
		int start_feature, int end_feature, int num_data){
	std::string line;
	std::fstream fp;
	fp.open(filename);

	arma::Mat<float> data = arma::Mat<float>(num_data, end_feature-start_feature,
			arma::fill::zeros);
	arma::Col<float> target = arma::Col<float>(num_data, arma::fill::zeros);

	for (int i = 0; i < num_data; i++) {
		std::getline(fp, line);
		if (line.back() != ' ') {
			line.push_back(' ');
		}
		// read target
		int cur = 0;
		target(i) = std::stoi(line.substr(0, line.find_first_of(' ', cur) - cur));
		cur = line.find_first_of(' ', cur) + 1;

		while (cur < line.length()) {
			int next_space = line.find_first_of(' ', cur);
			int next_colon = line.find_first_of(':', cur);
			int indx = std::stoi(line.substr(cur, next_colon-cur));
			float val = std::stod(line.substr(next_colon+1, next_space-next_colon-1));
			cur = next_space + 1;
			if (indx >= end_feature) {
				break;
			}
			else if (indx >= start_feature) {
				data(i, indx - start_feature) = val;
			}
		}
	}

	fp.close();
	return std::make_pair(data, target);
}
