#include <fstream>
#include <string>
#include <armadillo>

#include "read_libsvm.h"

using namespace arma;

// return: data (dxn), target(nx1)
std::pair<Mat<float>, Col<float>> ReadLibsvm(std::string filename,
		int start_feature, int end_feature, int num_data){
	std::string line;
	std::fstream fp;
	fp.open(filename);

	Mat<float> data = Mat<float>(end_feature-start_feature, num_data, fill::zeros);
	Col<float> target = Col<float>(num_data, fill::zeros);

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
				data(indx - start_feature, i) = val;
			}
		}
	}

	fp.close();
	return std::make_pair(data, target);
}

std::pair<SpMat<float>, Col<float>> ReadLibsvmSp(std::string filename,
		int start_feature, int end_feature, int num_data) {
	std::string line;
	std::fstream fp;
	fp.open(filename);

	Col<float> target = Col<float>(num_data, fill::zeros);

	std::vector<float> values;
	std::vector<long long unsigned int> row_idx, col_idx;

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
				values.push_back(val);
				row_idx.push_back(indx - start_feature);
				col_idx.push_back(i);
			}
		}
	}
	// batch insert data
	Mat<uword> locations(2, values.size());
	locations.row(0) = Row<uword>(row_idx);
	locations.row(1) = Row<uword>(col_idx);
	SpMat<float> data = SpMat<float>(locations, Col<float>(values), 
			end_feature - start_feature, num_data);

	fp.close();
	return std::make_pair(data, target);
}
