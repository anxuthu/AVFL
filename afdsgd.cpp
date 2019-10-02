#include <iostream>
#include <chrono>
#include <cassert>
#include <armadillo>
#include <string>
#include <utility>
#include <iomanip>
#include <cstdlib>

#include <mpi.h>

#include <thread>
#include <mutex>

#include <gflags/gflags.h>

#include "src/read_libsvm.h"
#include "src/logistic.h"

DEFINE_int32(t, 10, "svrg outer iterations");
DEFINE_int32(m, 19996, "svrg inner iterations");
DEFINE_double(reg, 1e-4, "regularization coefficient");
DEFINE_double(eta, 1, "step size");

DEFINE_string(train_data, "/export/UserData/an/real-sim", "training data path");
DEFINE_string(test_data, "/export/UserData/an/real-sim.t", "training data path");
DEFINE_int32(d, 20958, "number of features");
DEFINE_int32(n_train, 57848, "number of training instances");
DEFINE_int32(n_test, 14461, "number of testing instances");

DEFINE_int32(save_interval, 1, "save every #iter");
DEFINE_int32(save_num, 20, "total save num");

DEFINE_int32(seed, 1234, "Random seed.");

using namespace std;
using namespace arma;

mutex wl_mutex;

void train(int rank, int size, Mat<float> &xl, Col<float> &yl, Col<float> &wl,
		int steps, float eta, float reg) {
	MPI_Status status;
	for (int i = 0; i < steps; i++) {
		int idx = rand() % xl.n_rows;
		Col<float> total_prod(1, fill::zeros), prod(1, fill::zeros);
		for (int r = 0; r < size; r++) {
			MPI_Send(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
			MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, r, 1, MPI_COMM_WORLD,
					&status);
			total_prod += prod;
		}

		// compute stochastic gradient
		Col<float> grad = logistic_grad(total_prod, xl.row(idx), yl.row(idx));
		{
			lock_guard<mutex> lock(wl_mutex);
			wl = wl - eta * (grad + reg * wl);
		}
	}
	cout << rank << endl;

	// stop listener
	int idx = -1;
	for (int r = 0; r < size; r++) {
		MPI_Send(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
	}
}

void listener(int rank, int size, Mat<float> &xl, Col<float> &wl) {
	int source, idx, finished_workers = 0;
	MPI_Status status;
	while (1) {
		MPI_Recv(&idx, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
				&status);
		source = status.MPI_SOURCE;

		if (idx == -1) {
			finished_workers ++;
			if (finished_workers == size) {
				break;
			}
			else {
				continue;
			}
		}

		// local prod
		Col<float> prod(1, fill::zeros);
		{
			lock_guard<mutex> lock(wl_mutex);
			prod = xl.row(idx) * wl;
		}
		MPI_Send(prod.begin(), prod.size(), MPI_FLOAT, source, 1, MPI_COMM_WORLD);
	}
}

void test() {}

int main(int argc, char* argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	arma_rng::set_seed(FLAGS_seed);
	srand(FLAGS_seed);

	int rank, size, provided, thread_level = MPI_THREAD_MULTIPLE;
	MPI_Status status;
	
	MPI_Init_thread(&argc, &argv, thread_level, &provided);
	assert(thread_level <= provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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

	// initialize and load data
	std::pair<Mat<float>, Col<float>> data = ReadLibsvm(
			FLAGS_train_data, start_feature, end_feature, FLAGS_n_train);
	Mat<float> xl = std::get<0>(data);
	Col<float> yl = std::get<1>(data);
	Col<float> wl(xl.n_cols, fill::zeros);
	std::cout << "Process " << rank << " load data size " << xl.n_rows
		<< "x" << xl.n_cols << " and " << xl.n_rows << "x1" << std::endl;

	// start
	thread t1([&]{train(rank, size, xl, yl, wl, 1000, FLAGS_eta, FLAGS_reg);});
	thread t2([&]{listener(rank, size, xl, wl);});
	t1.join();
	t2.join();

	MPI_Finalize();
	return 0;
}
