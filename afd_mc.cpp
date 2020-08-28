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

#include <mpi.h>

#include <thread>
#include <mutex>

#include <gflags/gflags.h>

#include "src/read_libsvm.h"
#include "src/logistic_mc.h"
#include "src/sparse_ops_mc.h"

DEFINE_double(reg, 1e-4, "regularization coefficient");
DEFINE_double(eta, 0.1, "step size");

DEFINE_string(train_data, "/home/an/datasets/news20.scale", "training data path");
DEFINE_string(test_data, "/home/an/datasets/news20.t.scale", "training data path");
DEFINE_int32(d, 62061, "number of features");
DEFINE_int32(n_train, 15935, "number of training instances");
DEFINE_int32(n_test, 3993, "number of testing instances");
DEFINE_int32(n_class, 20, "number of classes (starting from 0 or 1)");

DEFINE_int32(method, 0, "0-sgd, 1-svrg, 2-saga");
DEFINE_int32(inner, 15935, "inner iteration");
DEFINE_int32(save_interval, 15935, "save every #iter");
DEFINE_int32(save_num, 1, "total save num");

DEFINE_int32(root, 0, "MPI root.");
DEFINE_int32(seed, 1234, "Random seed.");

DEFINE_int32(sync, 0, "Async or sync");
DEFINE_int32(max, 0, "");
DEFINE_double(comm_delay_fraction, 0, "communication delay (synthetic)");
DEFINE_double(delay_fraction, 0, "random delay of straggler");
DEFINE_int32(straggler, 1, "rank of the straggler");

using namespace std;
using namespace arma;
using namespace std::chrono;

mutex wl_mutex;


void listener(SpMat<float> &xl, Mat<float> &wl) {
	int size, source, idx, finished_workers = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	float comm_time = 0;

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
		Col<float> prod(FLAGS_n_class, fill::zeros);
		{
			lock_guard<mutex> lock(wl_mutex);
			Mul(xl.col(idx), wl, prod);
		}

		// comm delay
		//this_thread::sleep_for(nanoseconds(int(comm_time * FLAGS_comm_delay_fraction)));

		// send back result
		steady_clock::time_point comm_start = steady_clock::now();
		MPI_Send(prod.begin(), prod.size(), MPI_FLOAT, source, 1, MPI_COMM_WORLD);
		comm_time = duration_cast<duration<float>>(steady_clock::now() - comm_start).count() * 1e9;
	}
}


void sgd_train(SpMat<float>& xl, Col<float>& yl, Mat<float>& wl, Cube<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, comp_start, tmp;
	float comm=0, comp=0, sync=0;

	Col<float> total_prod(FLAGS_n_class, fill::zeros), prod(FLAGS_n_class, fill::zeros);

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
		// compute total prod with communication
		int idx = rand() % xl.n_cols;
		total_prod.fill(0);
		tmp = steady_clock::now();
		for (int r = 0; r < size; r++) {
			MPI_Isend(&idx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
		}
		for (int r = 0; r < size; r++) {
			MPI_Recv(prod.begin(), prod.size(), MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
			total_prod += prod;
		}
		comm += duration_cast<duration<float>>(steady_clock::now() - tmp).count();

		tmp = steady_clock::now();
		if (FLAGS_sync) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
		sync += duration_cast<duration<float>>(steady_clock::now() - tmp).count();

		comp_start = steady_clock::now(); // subtract communication time
		// compute stochastic gradient
		{
			lock_guard<mutex> lock(wl_mutex);
			LogisticMCL2Update(xl.col(idx), yl.row(idx), total_prod, wl, FLAGS_eta, FLAGS_reg);
		}

		// straggler
		if (rank == FLAGS_straggler) {
			float sleep_time = duration_cast<duration<float>>(
					steady_clock::now() - comp_start).count() * 1e9 * FLAGS_delay_fraction;
			this_thread::sleep_for(nanoseconds(int(sleep_time)));
		}
		comp += duration_cast<duration<float>>(steady_clock::now() - comp_start).count();

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.slice(to_save) = wl;
			elapsed(to_save) = duration_cast<duration<float>>(steady_clock::now() - start).count();
			cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
		}
	}
	cout << "comp " << comp << ", comm " << comm << ", sync " << sync << endl;
}


void svrg_train(SpMat<float>& xl, Col<float>& yl, Mat<float>& wl, Cube<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, iter_start;

	Col<float> total_prod(FLAGS_n_class, fill::zeros), prod(FLAGS_n_class, fill::zeros);
	Mat<float> total_full_prod(FLAGS_n_class, xl.n_cols, fill::zeros), 
		full_prod(FLAGS_n_class, xl.n_cols, fill::zeros);
	Mat <float> full_gradl(FLAGS_n_class, xl.n_rows, fill::zeros);

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
		LogisticMCGrad(xl, yl, total_full_prod, full_gradl);
		bool stop = false;
		cout << "Process " << rank << " outer iteration " << cur_outer << ". Full grad time: "
			<< duration_cast<duration<float>>(steady_clock::now() - start).count() << endl;

		// inner iteration
		int saved = cur_step / FLAGS_save_interval;
		float base = elapsed(saved);
		start = steady_clock::now();
		for (int i = 0; i < FLAGS_inner; ++i) {
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
			iter_start = steady_clock::now(); // subtract communication time

			// compute stochastic gradient
			{
				lock_guard<mutex> lock(wl_mutex);
				LogisticMCL2Update(full_gradl, xl.col(idx), yl.row(idx), total_full_prod.col(idx),
						total_prod, wl, FLAGS_eta, FLAGS_reg);
			}

			// straggler
			if (rank == FLAGS_straggler) {
				float sleep_time = duration_cast<duration<float>>(
						steady_clock::now() - iter_start).count() * 1e9 * FLAGS_delay_fraction;
				this_thread::sleep_for(nanoseconds(int(sleep_time)));
			}

			// save
			cur_step ++;
			if (cur_step % FLAGS_save_interval == 0) {
				int to_save = cur_step / FLAGS_save_interval;
				wl_bk.slice(to_save) = wl;
				elapsed(to_save) = duration_cast<duration<float>>(
						steady_clock::now() - start).count() + base;
				cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
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


void saga_train(SpMat<float>& xl, Col<float>& yl, Mat<float>& wl, Cube<float>& wl_bk,
		Col<float>& elapsed) {
	int rank, size;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	steady_clock::time_point start, iter_start;

	Col<float> total_prod(FLAGS_n_class, fill::zeros), prod(FLAGS_n_class, fill::zeros);
	Mat<float> total_full_prod(FLAGS_n_class, xl.n_cols, fill::zeros),
		full_prod(FLAGS_n_class, xl.n_cols, fill::zeros);
	Cube<float> hist_gradl(FLAGS_n_class, xl.n_rows, xl.n_cols, fill::zeros);
	Mat<float> hist_gradl_avg(FLAGS_n_class, xl.n_rows, fill::zeros);
	Mat<float> gradl(FLAGS_n_class, xl.n_rows, fill::zeros);

	Mul(xl, wl, full_prod);
	MPI_Allreduce(full_prod.begin(), total_full_prod.begin(), full_prod.size(),
			MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	LogisticMCGrad(xl, yl, total_full_prod, hist_gradl);
	LogisticMCGrad(xl, yl, total_full_prod, hist_gradl_avg);

	MPI_Barrier(MPI_COMM_WORLD);
	start = steady_clock::now();
	int steps = FLAGS_save_num * FLAGS_save_interval;
	for (int i = 0; i < steps; i++) {
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
		iter_start = steady_clock::now(); // subtract communication time

		// compute stochastic gradient
		{
			lock_guard<mutex> lock(wl_mutex);
			LogisticMCGrad(xl.col(idx), yl.row(idx), total_prod, gradl);
			wl *= 1 - FLAGS_eta * FLAGS_reg;
			wl -= FLAGS_eta * (gradl - hist_gradl.slice(idx) +  hist_gradl_avg);
			hist_gradl_avg += (gradl - hist_gradl.slice(idx)) / xl.n_cols;
			hist_gradl.slice(idx) = gradl;
		}

		// straggler
		if (rank == FLAGS_straggler) {
			float sleep_time = duration_cast<duration<float>>(
					steady_clock::now() - iter_start).count() * 1e9 * FLAGS_delay_fraction;
			this_thread::sleep_for(nanoseconds(int(sleep_time)));
		}

		// save
		if ((i + 1) % FLAGS_save_interval == 0) {
			int to_save = (i + 1) / FLAGS_save_interval;
			wl_bk.slice(to_save) = wl;
			elapsed(to_save) = duration_cast<duration<float>>(steady_clock::now() - start).count();
			cout << "Process " << rank << " saved num: " << to_save << " time: " << elapsed(to_save) << endl;
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
	Col<float> yl = get<1>(data) - min(get<1>(data));
	Mat<float> wl(FLAGS_n_class, xl.n_rows, fill::zeros);

	Cube<float> wl_bk(FLAGS_n_class, xl.n_rows, FLAGS_save_num + 1, fill::zeros);
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
	Col<float> test_yl = get<1>(test_data) - min(get<1>(test_data));
	Mat<float> test_total_prod(FLAGS_n_class, test_xl.n_cols, fill::zeros);
	Mat<float> test_prod(FLAGS_n_class, test_xl.n_cols, fill::zeros);
	Mat<float> train_total_prod(FLAGS_n_class, xl.n_cols, fill::zeros);
	Mat<float> train_prod(FLAGS_n_class, xl.n_cols, fill::zeros);
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
		Mat<float> wl = wl_bk.slice(i);
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
			float test_obj, train_obj, test_acc, train_acc;
			test_obj = logistic_mc(test_total_prod, test_yl) + total_l2_reg;
			train_obj = logistic_mc(train_total_prod, yl) + total_l2_reg;
			test_acc = logistic_mc_acc(test_total_prod, test_yl);
			train_acc = logistic_mc_acc(train_total_prod, yl);
			cout << "Step: " << FLAGS_save_interval * i
				<< " Time: " << elapsed(i) << setprecision(20)
				<< " Train obj: " << train_obj << " Test obj: " << test_obj
				<< " Train acc: " << train_acc << " Test acc: " << test_acc << endl;
		}
	}

	if (rank == FLAGS_root) {
		cout << "Evaluation time: " << duration_cast<duration<float>>(
				steady_clock::now() - start).count() << endl;
	}

	MPI_Finalize();
	return 0;
}
