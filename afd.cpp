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
#include "src/lr.h"
#include "src/sparse_ops.h"

DEFINE_double(reg, 1e-4, "regularization coefficient");
DEFINE_double(eta, 0.1, "step size");

DEFINE_string(train_data, "/ihome/hhuang/anx6/datasets/rcv1_test_binary", "training data path");
DEFINE_string(test_data, "/ihome/hhuang/anx6/datasets/rcv1_train_binary", "training data path");
DEFINE_int32(d, 47236, "number of features");
DEFINE_int32(n_train, 677399, "number of training instances");
DEFINE_int32(n_test, 20242, "number of testing instances");

DEFINE_int32(model, 0, "0-logistic regression, 1-linear regression");
DEFINE_int32(method, 0, "0-sgd, 1-svrg, 2-saga");
DEFINE_int32(inner, 677399, "inner iteration");
DEFINE_int32(save_interval, 677399, "save every #iter");
DEFINE_int32(save_num, 1, "total save num");

DEFINE_int32(root, 0, "MPI root.");
DEFINE_int32(seed, 1234, "Random seed.");

DEFINE_int32(sync, 0, "Async or sync");
DEFINE_int32(max, 0, "");
DEFINE_double(delay_fraction, 0, "random delay");
DEFINE_int32(straggler, 0, "rank of the straggler");

using namespace std;
using namespace arma;
using namespace std::chrono;

mutex wl_mutex;


void listener(SpMat<float> &xl, Col<float> &wl) {
	int size, source, idx, finished_workers = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	while (1) {
		MPI_Recv(&idx, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
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

		// compute local prod
		Col<float> prod(1, fill::zeros);
		{
			lock_guard<mutex> lock(wl_mutex);
			Mul(xl.col(idx), wl, prod);
		}
		// send back result
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
	steady_clock::time_point start, iter_start;

	Col<float> total_prod(1, fill::zeros), prod(1, fill::zeros);

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
		iter_start = steady_clock::now();

		// compute total prod
		int idx = rand() % xl.n_cols;
		total_prod.fill(0);
		for (int r = 0; r < size; r++) {
			MPI_Isend(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
		}
		for (int r = 0; r < size; r++) {
			MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
			total_prod += prod;
		}
		if (FLAGS_sync) {
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// compute stochastic gradient
		{
			lock_guard<mutex> lock(wl_mutex);
			if (FLAGS_model == 0) {
				LogisticL2Update(xl.col(idx), yl.row(idx), total_prod, wl, FLAGS_eta, FLAGS_reg);
			}
			else if (FLAGS_model == 1) {
				LRL2Update(xl.col(idx), yl.row(idx), total_prod, wl, FLAGS_eta, FLAGS_reg);
			}
		}

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.col(to_save) = wl;
			elapsed(to_save) = duration_cast<duration<float>>(steady_clock::now() - start).count();
			cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
		}

		// straggler
		if (rank == FLAGS_straggler) {
			float sleep_time = duration_cast<duration<float>>(
					steady_clock::now() - iter_start).count() * 1e9 * FLAGS_delay_fraction;
			this_thread::sleep_for(nanoseconds(int(sleep_time)));
		}
	}
}


void svrg_train(SpMat<float>& xl, Col<float>& yl, Col<float>& wl, Mat<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, iter_start;

	Col<float> total_prod(1, fill::zeros), prod(1, fill::zeros);
	Col<float> total_full_prod(xl.n_cols, fill::zeros), full_prod(xl.n_cols, fill::zeros);
	Col<float> full_gradl(xl.n_rows, fill::zeros);

	MPI_Barrier(MPI_COMM_WORLD);
	int steps = FLAGS_save_num * FLAGS_save_interval;
	int cur_step = 0;
	int cur_outer = 0;

	while (1) {
		// outer iteration
		start = steady_clock::now();
		{
			lock_guard<mutex> lock(wl_mutex);
			Mul(xl, wl, full_prod);
		}
		MPI_Allreduce(full_prod.begin(), total_full_prod.begin(), full_prod.size(),
				MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		if (FLAGS_model == 0) {
			LogisticGrad(xl, yl, total_full_prod, full_gradl);
		}
		else if (FLAGS_model == 1) {
			LRGrad(xl, yl, total_full_prod, full_gradl);
		}
		bool stop = false;
		cout << "Process " << rank << " outer iteration " << cur_outer << ". Full grad time: "
			<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

		// inner iteration
		int saved = cur_step / FLAGS_save_interval;
		float base = elapsed(saved);
		start = steady_clock::now();
		for (int i = 0; i < FLAGS_inner; ++i) {
			iter_start = steady_clock::now();

			// compute total prod
			int idx = rand() % xl.n_cols;
			total_prod.fill(0);
			for (int r = 0; r < size; r++) {
				MPI_Isend(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
			}
			for (int r = 0; r < size; r++) {
				MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
				total_prod += prod;
			}
			if (FLAGS_sync) {
				MPI_Barrier(MPI_COMM_WORLD);
			}

			// compute stochastic gradient
			{
				lock_guard<mutex> lock(wl_mutex);
				if (FLAGS_model == 0) {
					LogisticL2Update(full_gradl, xl.col(idx), yl.row(idx), total_full_prod.row(idx),
						total_prod, wl, FLAGS_eta, FLAGS_reg);
				}
				else if (FLAGS_model == 1) {
					LRL2Update(full_gradl, xl.col(idx), yl.row(idx), total_full_prod.row(idx),
						total_prod, wl, FLAGS_eta, FLAGS_reg);
				}
			}

			// save
			cur_step ++;
			if (cur_step % FLAGS_save_interval == 0) {
				int to_save = cur_step / FLAGS_save_interval;
				wl_bk.col(to_save) = wl;
				elapsed(to_save) = duration_cast<duration<float>>(
						steady_clock::now() - start).count() + base;
				cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
			}

			// straggler
			if (rank == FLAGS_straggler) {
				float sleep_time = duration_cast<duration<float>>(
						steady_clock::now() - iter_start).count() * 1e9 * FLAGS_delay_fraction;
				this_thread::sleep_for(nanoseconds(int(sleep_time)));
			}

			// finished
			if (cur_step  == steps) {
				stop = true;
				break;
			}
		}

		if (stop) {
			break;
		}
		cur_outer ++;
	}
}


void saga_train(SpMat<float>& xl, Col<float>& yl, Col<float>& wl, Mat<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, iter_start;

	Col<float> total_prod(1, fill::zeros), prod(1, fill::zeros);
	Col<float> total_full_prod(xl.n_cols, fill::zeros), full_prod(xl.n_cols, fill::zeros);
	Mat<float> hist_gradl(xl.n_rows, xl.n_cols, fill::zeros);
	Col<float> hist_gradl_avg(xl.n_rows, fill::zeros);
	Col<float> gradl(xl.n_rows, fill::zeros);

	Mul(xl, wl, full_prod);
	MPI_Allreduce(full_prod.begin(), total_full_prod.begin(), full_prod.size(),
			MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	if (FLAGS_model == 0) {
		LogisticGrad(xl, yl, total_full_prod, hist_gradl);
		LogisticGrad(xl, yl, total_full_prod, hist_gradl_avg);
	}
	else if (FLAGS_model == 1) {
		LRGrad(xl, yl, total_full_prod, hist_gradl);
		LRGrad(xl, yl, total_full_prod, hist_gradl_avg);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
		iter_start = steady_clock::now();

		// compute total prod
		int idx = rand() % xl.n_cols;
		total_prod.fill(0);
		for (int r = 0; r < size; r++) {
			MPI_Isend(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
		}
		for (int r = 0; r < size; r++) {
			MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
			total_prod += prod;
		}
		if (FLAGS_sync) {
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// compute stochastic gradient
		{
			lock_guard<mutex> lock(wl_mutex);
			if (FLAGS_model == 0) {
				LogisticGrad(xl.col(idx), yl.row(idx), total_prod, gradl);
			}
			else if (FLAGS_model == 1) {
				LRGrad(xl.col(idx), yl.row(idx), total_prod, gradl);
			}
			gradl += FLAGS_reg * wl;
			wl -= FLAGS_eta * (gradl - hist_gradl.col(idx) +  hist_gradl_avg);
			hist_gradl_avg += (gradl - hist_gradl.col(idx)) / xl.n_cols;
			hist_gradl.col(idx) = gradl;
		}

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.col(to_save) = wl;
			elapsed(to_save) = duration_cast<duration<float>>(steady_clock::now() - start).count();
			cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
		}

		// straggler
		if (rank == FLAGS_straggler) {
			float sleep_time = duration_cast<duration<float>>(
					steady_clock::now() - iter_start).count() * 1e9 * FLAGS_delay_fraction;
			this_thread::sleep_for(nanoseconds(int(sleep_time)));
		}
	}
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
		<< " OMP max threads: " << omp_get_max_threads()
		<< " Train file: " << FLAGS_train_data << " Test file: " << FLAGS_test_data
		<< " method: " << FLAGS_method << " sync: " << FLAGS_sync
		<< " eta: " << FLAGS_eta << " straggler delay fraction: " << FLAGS_delay_fraction << endl;

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
	Col<float> yl = get<1>(data);
	Col<float> wl(xl.n_rows, fill::zeros);

	Mat<float> wl_bk(xl.n_rows, FLAGS_save_num + 1, fill::zeros);
	Col<float> elapsed(FLAGS_save_num + 1, fill::zeros);
	cout << "Process " << rank << " load train_data size " << xl.n_rows << "x" << xl.n_cols
		<< " and " << xl.n_cols << "x1. Time: "
		<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

	// start listener
	thread t([&]{listener(xl, wl);});

	/****** Training ******/
	if (FLAGS_method == 0) {
		sgd_train(xl, yl, wl, wl_bk, elapsed);
	}
	else if (FLAGS_method == 1) {
		svrg_train(xl, yl, wl, wl_bk, elapsed);
	}
	else if (FLAGS_method == 2) {
		saga_train(xl, yl, wl, wl_bk, elapsed);
	}
	else {
		assert(false);
	}

	// stop listener
	int idx = -1;
	for (int r = 0; r < size; r++) {
		MPI_Send(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
	}
	t.join();

	/****** Testing ******/
	start = steady_clock::now();
	pair<SpMat<float>, Col<float>> test_data = ReadLibsvmSp(
			FLAGS_test_data, start_feature, end_feature, FLAGS_n_test);
	SpMat<float> test_xl = get<0>(test_data);
	Col<float> test_yl = get<1>(test_data);
	Col<float> test_total_prod(test_xl.n_cols, fill::zeros);
	Col<float> test_prod(test_xl.n_cols, fill::zeros);
	Col<float> train_total_prod(xl.n_cols, fill::zeros);
	Col<float> train_prod(xl.n_cols, fill::zeros);
	Col<float> new_elapsed(FLAGS_save_num + 1, fill::zeros);
	cout << "Process " << rank << " load test_data size " << test_xl.n_rows
		<< "x" << test_xl.n_cols << " and " << test_xl.n_cols << "x1. Time: "
		<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

	start = steady_clock::now();
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
		Mul(test_xl, wl, test_prod);
		MPI_Reduce(test_prod.begin(), test_total_prod.begin(), test_prod.size(),
				MPI_FLOAT, MPI_SUM, FLAGS_root, MPI_COMM_WORLD);

		Mul(xl, wl, train_prod);
		MPI_Reduce(train_prod.begin(), train_total_prod.begin(), train_prod.size(),
				MPI_FLOAT, MPI_SUM, FLAGS_root, MPI_COMM_WORLD);
		float total_l2_reg, l2_reg = 1.0 / 2 * FLAGS_reg * accu(pow(wl, 2));
		MPI_Reduce(&l2_reg, &total_l2_reg, 1, MPI_FLOAT, MPI_SUM, FLAGS_root,
				MPI_COMM_WORLD);

		if (rank == FLAGS_root) {
			float test_obj, train_obj;
			if (FLAGS_model == 0) {
				test_obj = logistic(test_total_prod, test_yl) + total_l2_reg;
				train_obj = logistic(train_total_prod, yl) + total_l2_reg;
			}
			else if (FLAGS_model == 1) {
				test_obj = LR(test_total_prod, test_yl) + total_l2_reg;
				train_obj = LR(train_total_prod, yl) + total_l2_reg;
			}
			cout << "Step: " << FLAGS_save_interval * i
				<< " Time: " << elapsed(i) << setprecision(20)
				<< " Train obj: " << train_obj << " Test obj: " << test_obj << endl;
		}
	}

	if (rank == FLAGS_root) {
		cout << "Evaluation time: " << duration_cast<duration<float>>(
				steady_clock::now() - start).count() << endl;
	}

	MPI_Finalize();
	return 0;
}
