/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/perf_counter.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __PERF_COUNTER_H__
#define __PERF_COUNTER_H__

#include <vector>

class PerformanceCounter {
	public:
		enum OperationTypes
		{
			SVD,
			QR,
            GEMM,
			TotalOps
		};

	private:
		// Constructors
		PerformanceCounter()
		{
			for(int i = 0; i < TotalOps; i++)
				gops[i] = op_time[i] = 0;
		}
		PerformanceCounter(PerformanceCounter const& copy);
		PerformanceCounter& operator=(PerformanceCounter const& copy);

		// Performance data
		double gops[TotalOps], op_time[TotalOps];
	public:
		static PerformanceCounter& get()
		{
			static PerformanceCounter instance;
			return instance;
		}

		static void clearCounters()
		{
			for(int i = 0; i < TotalOps; i++)
			{
				setOpCount((OperationTypes)i, 0);
				setOpTime((OperationTypes)i, 0);
			}
		}

		// GigaOp counts
		static double getOpCount(OperationTypes type)
		{
			return get().gops[type];
		}
        static void setOpCount(OperationTypes type, double val)
		{
			get().gops[type] = val;
		}
        static void addOpCount(OperationTypes type, double val)
		{
			get().gops[type] += val;
		}

		// Time
		static double getOpTime(OperationTypes type)
		{
			return get().op_time[type];
		}
        static void setOpTime(OperationTypes type, double time)
		{
			get().op_time[type] = time;
		}
        static void addOpTime(OperationTypes type, double time)
		{
			get().op_time[type] += time;
		}
};

class HLibProfile {
	public:
		enum HgemvProfile {
			HGEMV_UPSWEEP = 0,
			HGEMV_MULT,
			HGEMV_DOWNSWEEP,
			HGEMV_DENSE,
			HGEMV_TOTAL
		};

		enum HorthogProfile {
			HORTHOG_BASIS_LEAVES = HGEMV_TOTAL,
			HORTHOG_UPSWEEP,
			HORTHOG_STITCH,
			HORTHOG_PROJECTION,
			HORTHOG_TOTAL
		};

		enum HcompressProfile {
			HCOMPRESS_BASIS_GEN = HORTHOG_TOTAL,
			HCOMPRESS_TRUNCATE_BASIS,
			HCOMPRESS_PROJECTION,
			HCOMPRESS_STITCH,
			HCOMPRESS_TOTAL,
			HLibProfileCount = HCOMPRESS_TOTAL
		};

		template<class HLibProfilePhase>
		static void addRunT(HLibProfilePhase phase, double perf_metric, double perf_time)
		{
			get().operation_perf[phase].push_back(perf_metric);
			get().operation_time[phase].push_back(perf_time);
		}

		template<class HLibProfilePhase>
		static void getPhasePerformanceT(HLibProfilePhase phase, double& avg_metric, double& avg_time, double& avg_perf, double& perf_std_dev)
		{
			std::vector<double>& phase_perf = get().operation_perf[phase];
			std::vector<double>& phase_time = get().operation_time[phase];

			avg_perf = avg_metric = avg_time = perf_std_dev = 0;
			int total_runs = phase_perf.size();

			if(total_runs == 0) return;

			// Skip first warmup run if possible
			int run_start = (total_runs > 1 ? 1 : 0);

			for(int run = run_start; run < total_runs; run++)
			{
				avg_metric += phase_perf[run];
				avg_time += phase_time[run];
				avg_perf += phase_perf[run] / phase_time[run];
			}

			avg_metric /= (total_runs - run_start);
			avg_time /= (total_runs - run_start);
			avg_perf /= (total_runs - run_start);

			for(int run = run_start; run < total_runs; run++)
				perf_std_dev += (avg_perf - phase_perf[run] / phase_time[run]) * (avg_perf - phase_perf[run] / phase_time[run]);
			perf_std_dev = sqrt(perf_std_dev / (total_runs - run_start));
		}

		static double getRunTotalMetric(int run, int phase_start, int phase_end)
		{
			double total = 0;

			for(int phase = phase_start; phase <= phase_end; phase++)
			{
				std::vector<double>& phase_perf = get().operation_perf[phase];
				if(run >= phase_perf.size()) continue;
				total += phase_perf[run];
			}

			return total;
		}

		static double getRunTotalTime(int run, int phase_start, int phase_end)
		{
			double total = 0;

			for(int phase = phase_start; phase <= phase_end; phase++)
			{
				std::vector<double>& phase_time = get().operation_time[phase];
				if(run >= phase_time.size()) continue;
				total += phase_time[run];
			}

			return total;
		}

		static void getOperationPerformance(int phase_start, int phase_end, double& avg_metric, double& avg_time, double& avg_perf, double& perf_std_dev)
		{
			avg_perf = avg_metric = avg_time = 0;

			int total_runs = 0;
			for(int phase = phase_start; phase <= phase_end; phase++)
				if(total_runs < get().operation_perf[phase_start].size())
					total_runs = get().operation_perf[phase_start].size();

			if(total_runs == 0) return;

			// Skip first warmup run if possible
			int run_start = (total_runs > 1 ? 1 : 0);

			for(int run = run_start; run < total_runs; run++)
			{
				double total_metric = getRunTotalMetric(run, phase_start, phase_end);
				double total_time   = getRunTotalTime(run, phase_start, phase_end);

				avg_metric += total_metric;
				avg_time   += total_time;
				avg_perf   += total_metric / total_time;
				// printf("Run %d performance was %f\n", run, total_metric / total_time);
			}
			avg_metric /= (total_runs - run_start);
			avg_time /= (total_runs - run_start);
			avg_perf /= (total_runs - run_start);

			for(int run = run_start; run < total_runs; run++)
			{
				double total_metric = getRunTotalMetric(run, phase_start, phase_end);
				double total_time   = getRunTotalTime(run, phase_start, phase_end);
				perf_std_dev += (avg_perf - total_metric / total_time) * (avg_perf - total_metric / total_time);
			}
			perf_std_dev = sqrt(perf_std_dev / (total_runs - run_start));
		}
	private:
		std::vector<double> operation_time[HLibProfileCount];
		std::vector<double> operation_perf[HLibProfileCount];

		HLibProfile()
		{reset_counters();}

		void reset_counters()
		{
			for(int i = 0; i < HLibProfileCount; i++)
			{
				operation_time[i].clear();
				operation_perf[i].clear();
			}
		}
	public:
		static HLibProfile& get()
		{
			static HLibProfile instance;
			return instance;
		}

		static void addRun(HgemvProfile phase, double gbytes, double perf_time) { addRunT(phase, gbytes, perf_time); }
		static void addRun(HorthogProfile phase, double gops, double perf_time) { addRunT(phase, gops, perf_time);   }
		static void addRun(HcompressProfile phase, double gops, double perf_time) { addRunT(phase, gops, perf_time);   }

		static void getPhasePerformance(HgemvProfile phase, double& avg_gbs, double& avg_time, double& avg_perf, double& perf_std_dev) { getPhasePerformanceT(phase, avg_gbs, avg_time, avg_perf, perf_std_dev); }
		static void getPhasePerformance(HorthogProfile phase, double& avg_gops, double& avg_time, double& avg_perf, double& perf_std_dev) { getPhasePerformanceT(phase,  avg_gops, avg_time, avg_perf, perf_std_dev);   }
		static void getPhasePerformance(HcompressProfile phase, double& avg_gops, double& avg_time, double& avg_perf, double& perf_std_dev) { getPhasePerformanceT(phase,  avg_gops, avg_time, avg_perf, perf_std_dev);   }

		static void getHgemvPerf(double& avg_gbs, double& avg_time, double& avg_perf, double& perf_std_dev) { getOperationPerformance(HGEMV_UPSWEEP, HGEMV_DENSE, avg_gbs, avg_time, avg_perf, perf_std_dev); }
		static void getHorthogPerf(double& avg_gops, double& avg_time, double& avg_perf, double& perf_std_dev) { getOperationPerformance(HORTHOG_BASIS_LEAVES, HORTHOG_PROJECTION, avg_gops, avg_time, avg_perf, perf_std_dev); }
		static void getHcompressPerf(double& avg_gops, double& avg_time, double& avg_perf, double& perf_std_dev) { getOperationPerformance(HCOMPRESS_BASIS_GEN, HCOMPRESS_STITCH, avg_gops, avg_time, avg_perf, perf_std_dev); }
};

#endif
