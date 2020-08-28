// multinomial logistic regression for multiclass classification
#include <iostream>
#include <chrono>
#include <cassert>
#include <armadillo>
#include <string>
#include <utility>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <math.h>

#include <mpi.h>

#include <thread>
#include <mutex>

#include <gflags/gflags.h>

#include "src/read_libsvm.h"
#include "src/sparse_ops_mc.h"

DEFINE_double(reg, 1e-4, "regularization coefficient");
DEFINE_double(eta, 0.1, "step size");

DEFINE_string(train_data, "./datasets/rcv1_test.multiclass", "training data path");
DEFINE_string(test_data, "./datasets/rcv1_train.multiclass", "training data path");
DEFINE_int32(d, 47236, "number of features");
DEFINE_int32(n_train, 518571, "number of training instances");
DEFINE_int32(n_test, 15564, "number of testing instances");
DEFINE_int32(n_class, 53, "number of classes (starting from 0 or 1)");

DEFINE_int32(save_interval, 20000, "save every #iter");
DEFINE_int32(save_num, 40, "total save num");

DEFINE_int32(root, 0, "MPI root.");
DEFINE_int32(seed, 1234, "Random seed.");

using namespace std;
using namespace arma;
using namespace std::chrono;


void sgd_train(SpMat<float>& xl, Col<float>& yl, Mat<float>& wl, Cube<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, comp_start, tmp;

	Col<float> prod(FLAGS_n_class, fill::zeros);
	Col<float> pred(FLAGS_n_class, fill::zeros), total_pred(FLAGS_n_class, fill::zeros);

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
		// compute total prod with communication
		int idx = rand() % xl.n_cols;
		Mul(xl.col(idx), wl, prod);

		//LogisticMCL2Update(xl.col(idx), yl.row(idx), pred, wl, FLAGS_eta, FLAGS_reg);
		// fdml server gathers, worker updates
		pred = exp(prod) / accu(exp(prod));
		MPI_Allreduce(pred.begin(), total_pred.begin(), pred.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		total_pred /= size;
		float fraction = pred(yl(idx)) / total_pred(yl(idx)) * size;//TODO
		LogisticMCL2Update(xl.col(idx), yl.row(idx), prod, wl, FLAGS_eta, FLAGS_reg, fraction);

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.slice(to_save) = wl;
			elapsed(to_save) = duration_cast<duration<float>>(steady_clock::now() - start).count();
			cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
		}
	}
}


pair<float, float> eval(SpMat<float>& xl, Col<float>& yl, Mat<float>& wl) {
	Mat<float> pred(FLAGS_n_class, xl.n_cols), total_pred(FLAGS_n_class, xl.n_cols);
	float obj = 0, acc = xl.n_cols;
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	Mul(xl, wl, pred);
	pred.each_col([](Col<float>& col){
			col = exp(col);
			col /= accu(col);
	});
	MPI_Allreduce(pred.begin(), total_pred.begin(), pred.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	total_pred /= size;

	// objective (-plog(p)) and acc (argmax)
	for (int i = 0; i < xl.n_cols; i++) {
		float p = total_pred((yl(i)), i);
		obj += - log(p);

		for (int j = 0; j < FLAGS_n_class; j++) {
			if (total_pred(j, i) >= p && j != yl(i)) {
				acc -= 1;
				break;
			}
		}
	}
	return make_pair(obj / xl.n_cols, acc / xl.n_cols);
}


int main(int argc, char* argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	arma_rng::set_seed(FLAGS_seed);
	srand(FLAGS_seed);
	arma_version ver;

	int rank, size, provided, thread_level = MPI_THREAD_MULTIPLE;
	steady_clock::time_point start;
	MPI_Init_thread(&argc, &argv, thread_level, &provided);
	assert(thread_level <= provided && "MPI_THREAD_MULTIPLE not supported!");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	cout << "Process " << rank << " ARMA version: " << ver.as_string()
		<< " Train file: " << FLAGS_train_data << " Test file: " << FLAGS_test_data
		<< " eta: " << FLAGS_eta << endl;

	// sizes
	vector<int> feature_sizes(size, FLAGS_d / size);
	for (int i = 0; i < FLAGS_d % size; i++) {
		feature_sizes[i] += 1;
	}
	int start_feature = 0, end_feature = 0;
	for (int i = 0; i < rank; i++) {
		start_feature += feature_sizes[i];
	}
	end_feature = start_feature + feature_sizes[rank];

	// initialize weight and load data: xl (dlxn), yl (nx1)
	start = steady_clock::now();
	pair<SpMat<float>, Col<float>> data = ReadLibsvmSp(
			FLAGS_train_data, start_feature, end_feature, FLAGS_n_train);
	SpMat<float> xl = get<0>(data);
	Col<float> yl = get<1>(data) - min(get<1>(data));
	Mat<float> wl(FLAGS_n_class, xl.n_rows, fill::zeros);

	Cube<float> wl_bk(FLAGS_n_class, xl.n_rows, FLAGS_save_num + 1, fill::zeros);
	Col<float> elapsed(FLAGS_save_num + 1, fill::zeros);
	cout << "Process " << rank << " load train_data size " << xl.n_rows << "x" << xl.n_cols
		<< " and " << xl.n_cols << "x1. Time: "
		<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

	/****** Training ******/
	sgd_train(xl, yl, wl, wl_bk, elapsed);

	/****** Testing ******/
	start = steady_clock::now();
	pair<SpMat<float>, Col<float>> test_data = ReadLibsvmSp(
			FLAGS_test_data, start_feature, end_feature, FLAGS_n_test);
	SpMat<float> test_xl = get<0>(test_data);
	Col<float> test_yl = get<1>(test_data) - min(get<1>(test_data));
	Col<float> new_elapsed(FLAGS_save_num + 1, fill::zeros);
	cout << "Process " << rank << " load test_data size " << test_xl.n_rows
		<< "x" << test_xl.n_cols << " and " << test_xl.n_cols << "x1. Time: "
		<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

	start = steady_clock::now();
	MPI_Reduce(elapsed.begin(), new_elapsed.begin(), elapsed.size(), MPI_FLOAT,
			MPI_MIN, FLAGS_root, MPI_COMM_WORLD);
	elapsed = new_elapsed;

	for (int i = 0; i < FLAGS_save_num + 1; i++) {
		Mat<float> wl = wl_bk.slice(i);
		float total_l2_reg, l2_reg = 1.0 / 2 * FLAGS_reg * accu(pow(wl, 2));
		MPI_Reduce(&l2_reg, &total_l2_reg, 1, MPI_FLOAT, MPI_SUM, FLAGS_root, MPI_COMM_WORLD);

		pair<float, float> train_res = eval(xl, yl, wl);
		pair<float, float> test_res = eval(test_xl, test_yl, wl);
		if (rank == FLAGS_root) {
			cout << "Step: " << FLAGS_save_interval * i
				<< " Time: " << elapsed(i) << setprecision(20)
				<< " Train obj: " << get<0>(train_res) << " Test obj: " << get<0>(test_res) 
				<< " Train acc: " << get<1>(train_res) << " Test acc: " << get<1>(test_res) << endl;
		}
	}

	if (rank == FLAGS_root) {
		cout << "Evaluation time: " << duration_cast<duration<float>>(
				steady_clock::now() - start).count() << endl;
	}

	MPI_Finalize();
	return 0;
}
