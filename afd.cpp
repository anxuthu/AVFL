#include <iostream>
#include <chrono>
#include <cassert>
#include <armadillo>
#include <string>
#include <utility>
#include <iomanip>
#include <cstdlib>
#include <random>

#include <mpi.h>

#include <thread>
#include <mutex>

#include <gflags/gflags.h>

#include "src/read_libsvm.h"
#include "src/logistic.h"

DEFINE_double(reg, 1e-4, "regularization coefficient");
DEFINE_double(eta, 1, "step size");

DEFINE_string(train_data, "/export/UserData/an/real-sim", "training data path");
DEFINE_string(test_data, "/export/UserData/an/real-sim.t", "training data path");
DEFINE_int32(d, 20958, "number of features");
DEFINE_int32(n_train, 57848, "number of training instances");
DEFINE_int32(n_test, 14461, "number of testing instances");

DEFINE_int32(save_interval, 50, "save every #iter");
DEFINE_int32(save_num, 20, "total save num");

DEFINE_int32(root, 0, "MPI root.");
DEFINE_int32(seed, 1234, "Random seed.");

DEFINE_int32(sync, 0, "Async or sync");
DEFINE_int32(max, 0, "");
DEFINE_double(delay, 0, "random delay");

using namespace std;
using namespace arma;
using namespace std::chrono;

mutex wl_mutex;

void listener(int size, SpMat<float> &xl, Col<float> &wl) {
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
			prod = xl.col(idx).t() * wl;
		}
		MPI_Send(prod.begin(), prod.size(), MPI_FLOAT, source, 1, MPI_COMM_WORLD);
	}
}


void sgd_train(SpMat<float>& xl, Col<float>& yl, Col<float>& wl, Mat<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	Col<float> grad(xl.n_rows, fill::zeros);
	Col<float> total_prod(1, fill::zeros), prod(1, fill::zeros);

	steady_clock::time_point start, cur;
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0, FLAGS_delay);

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
		int delay = distribution(generator) * 1000000;
		this_thread::sleep_for (microseconds(delay));

		// compute total prod
		int idx = rand() % xl.n_cols;
		total_prod.fill(0);
		for (int r = 0; r < size; r++) {
			MPI_Isend(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
		}
		for (int r = 0; r < size; r++) {
			MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, MPI_ANY_SOURCE, 1,
					MPI_COMM_WORLD, &status);
			total_prod += prod;
		}
		if (FLAGS_sync) {
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// compute stochastic gradient
		grad = logistic_grad(total_prod, xl.col(idx), yl.row(idx));
		{
			lock_guard<mutex> lock(wl_mutex);
			wl = wl - FLAGS_eta * (grad + FLAGS_reg * wl);
		}

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.col(to_save) = wl;
			cur = steady_clock::now();
			duration<float> elapsed_sec = duration_cast<duration<float>>(
					cur - start);
			elapsed(to_save) = elapsed_sec.count();
            cout << "Process " << rank << " saved num: " << to_save << endl;
		}
	}
}


void svrg_train(SpMat<float>& xl, Col<float>& yl, Col<float>& wl, Col<float>& wl_bk,
		Col<float>& elapsed) {}


void saga_train(SpMat<float>& xl, Col<float>& yl, Col<float>& wl, Col<float>& wl_bk,
		Col<float>& elapsed) {}


int main(int argc, char* argv[]) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	arma_rng::set_seed(FLAGS_seed);
	srand(FLAGS_seed);
	arma_version ver;
	cout << "ARMA version: " << ver.as_string() << endl;

	int rank, size, provided, thread_level = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, thread_level, &provided);
	assert(thread_level <= provided && "MPI_THREAD_MULTIPLE not supported!");
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

	// initialize weight and load data: xl (dlxn), yl (nx1)
	pair<SpMat<float>, Col<float>> data = ReadLibsvmSp(
			FLAGS_train_data, start_feature, end_feature, FLAGS_n_train);
	SpMat<float> xl = get<0>(data);
	Col<float> yl = get<1>(data);
	Col<float> wl(xl.n_rows, fill::zeros);

	Mat<float> wl_bk(xl.n_rows, FLAGS_save_num + 1, fill::zeros);
	Col<float> elapsed(FLAGS_save_num + 1, fill::zeros);
	cout << "Process " << rank << " load data size " << xl.n_rows
		<< "x" << xl.n_cols << " and " << xl.n_cols << "x1" << endl;

	// start listener
	thread t([&]{listener(size, xl, wl);});

	/*** Training ***/
	sgd_train(xl, yl, wl, wl_bk, elapsed);

	// stop listener
	int idx = -1;
	for (int r = 0; r < size; r++) {
		MPI_Send(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
	}
	t.join();

	/*** Testing ***/
    cout << "Process " << rank << " start evaluation." << endl;
	pair<SpMat<float>, Col<float>> test_data = ReadLibsvmSp(
			FLAGS_test_data, start_feature, end_feature, FLAGS_n_test);
	SpMat<float> test_xl = get<0>(test_data);
	Col<float> test_yl = get<1>(test_data);
	Col<float> test_total_prod(test_xl.n_cols, fill::zeros);
	Col<float> test_prod(test_xl.n_cols, fill::zeros);
	Col<float> train_total_prod(xl.n_cols, fill::zeros);
	Col<float> train_prod(xl.n_cols, fill::zeros);
	Col<float> new_elapsed(FLAGS_save_num + 1, fill::zeros);

	if (FLAGS_max) {
		MPI_Reduce(elapsed.begin(), new_elapsed.begin(), elapsed.size(), MPI_FLOAT,
				MPI_MAX, FLAGS_root, MPI_COMM_WORLD);
		elapsed = new_elapsed;
	}
	else {
		MPI_Reduce(elapsed.begin(), new_elapsed.begin(), elapsed.size(), MPI_FLOAT,
				MPI_MIN, FLAGS_root, MPI_COMM_WORLD);
		elapsed = new_elapsed;
	}

	for (int i = 0; i < FLAGS_save_num + 1; i++) {
		Col<float> wl = wl_bk.col(i);
		test_prod = test_xl.t() * wl;
		MPI_Reduce(test_prod.begin(), test_total_prod.begin(), test_prod.size(),
				MPI_FLOAT, MPI_SUM, FLAGS_root, MPI_COMM_WORLD);
		train_prod = xl.t() * wl;
		MPI_Reduce(train_prod.begin(), train_total_prod.begin(), train_prod.size(),
				MPI_FLOAT, MPI_SUM, FLAGS_root, MPI_COMM_WORLD);
		if (rank == FLAGS_root) {
			float test_obj = logistic(test_total_prod, test_yl);
			float train_obj = logistic(train_total_prod, yl);
			cout << "Step: " << FLAGS_save_interval * i << " Time: " << elapsed(i)
				<< setprecision(15)
				<< " Train obj: " << train_obj << " Test obj: " << test_obj << endl;
		}
	}

	MPI_Finalize();
	return 0;
}
