#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include "Point.h"
#include "IOUtil.h"
#include "RMSUtils.h"
#include "MinHappiness.h"
#include "Greedy.h"
#include "ThreeSieves.h"
#include "Preemption.h"
#include "GreedyAT.h"
#include "SieveOnline.h"
#include "ThresholdOnline.h"
using namespace std;

const char *FilePath_0 = "./dataset/real/Football_3d/Football_3d_normalized.txt";

ofstream outfile, outfile_fval, outfile_time, outfile_maxn;

double evaluate_singleton_m(vector<Point> const &DataSet, vector<UtilityFunction> FC, MHR &mhr)
{
    vector<double> mhr_singleton_p(DataSet.size());
    for (size_t i = 0; i < DataSet.size(); ++i)
    {
        double mhr_p = mhr.operator()({DataSet[i]}, FC);
        mhr_singleton_p[i] = mhr_p;
    }
    double max_mhr_singleton_p = *max_element(mhr_singleton_p.begin(), mhr_singleton_p.end());
    return max_mhr_singleton_p;
}

auto evaluate_optimizer(SubsetSelectionAlgorithm &opt, vector<Point> &DataSet)
{
    auto start = chrono::steady_clock::now();
    opt.fit(DataSet);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> runtime_seconds = end - start;
    double fval = opt.get_fval();
    // return make_tuple(fval, runtime_seconds.count(), opt.solution.size(), opt.get_maxn());
    return make_tuple(fval, runtime_seconds.count(), opt.solution.size());
}

int main(int argc, char **argv)
{
    vector<string> FilePaths = {
                                "../dataset/real/Football_3d/Football_3d_normalized.txt",
                                "../dataset/real/Tweet_7d/Tweet_7d_normalized.txt",
                                "../dataset/real/Weather_15d/Weather_15d_normalized.txt",
                                "../dataset/synthetic/IND_3_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_3_100_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_3_1000_normalized.txt",
                                "../dataset/synthetic/Anti-Cor_3_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_3_100000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_3_1000000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_3_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_4_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_5_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_6_10000_normalized.txt",
                                // "./dataset/synthetic/Anti-Cor_7_10000_normalized.txt",
                                };
    for (size_t count = 0; count < FilePaths.size(); ++count)
    {
        size_t dim;
        vector<Point> dataP;
        IOUtil::read_input_points(FilePaths[count].c_str(), dim, dataP);

        /* Read Dataset */
        cout << "Reading data..." << endl;
        cout << "The size of Dataset is " << dataP.size() << endl;
        cout << "The dimension of Dataset is " << dim << endl;

        /* sample ndir utility functions and initialize */
        vector<UtilityFunction> FunctionClass;
        size_t ndir = 1000;
        RMSUtils::get_random_utility_functions(1.0, dim, ndir, FunctionClass, true);
        cout << "The size of FunctionClass is " << FunctionClass.size() << endl;

        for (size_t j = 0; j < FunctionClass.size(); ++j)
        {
            for (size_t i = 0; i < dataP.size(); ++i)
            {
                double f_tmp = FunctionClass[j].direction.dotP(dataP[i]);
                FunctionClass[j].fmax = f_tmp > FunctionClass[j].fmax ? f_tmp : FunctionClass[j].fmax;
            }
        }

        tuple<double, double, size_t> res;

        /* the cardinality constraint */
        vector<size_t> ks = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        // vector<size_t> ks = {30};

        vector<string> outfiles={
            // "./result/result_for_fig/mhr_varyP_ThresholdOnline_Tweet.txt",
            // "./result/result_for_fig/mhr_varyP_SieveOnline_Tweet.txt"
        };
        vector<string> outfiles_fval={            
            "./result/result_for_fig/Football/mhr_Football_fval.txt",
            "./result/result_for_fig/Tweet/mhr_Tweet_fval.txt",
            "./result/result_for_fig/Weather/mhr_Weather_fval.txt",
            "./result/result_for_fig/IND/mhr_IND_fval.txt",
            "../result/result_for_fig/Anti-Cor/mhr_Anti-Cor_fval.txt"
        };
        vector<string> outfiles_time={            
            "./result/result_for_fig/Football/mhr_Football_time.txt",
            "./result/result_for_fig/Tweet/mhr_Tweet_time.txt",
            "./result/result_for_fig/Weather/mhr_Weather_time.txt",
            "./result/result_for_fig/IND/mhr_IND_time.txt",
            "../result/result_for_fig/Anti-Cor/mhr_Anti-Cor_time.txt"
        };

        vector<string> outfiles_maxn = {
            // "./result/result_for_fig/Football/mhr_Football_maxn.txt",
            // "./result/result_for_fig/Tweet/mhr_Tweet_maxn.txt",
            // "./result/result_for_fig/Weather/mhr_Weather_maxn.txt",
            // "./result/result_for_fig/IND/mhr_IND_maxn.txt",
            // "../result/result_for_fig/Anti-Cor/mhr_Anti-Cor_maxn.txt"};

        for (size_t count_k = 0; count_k < ks.size(); ++count_k)
        {
            size_t k = ks[count_k];
            // outfile.open(outfiles[count], ios::out|ios::app);
            outfile_fval.open(outfiles_fval[count], ios::out|ios::app);
            outfile_time.open(outfiles_time[count], ios::out|ios::app);

            /* Define the truncated minimum happiness ratio function to be maximized */
            double tauHigh = 1.0, tauLow = 0.0, tau;
            double delta = 0.02;
            double fval_pre = 0.0;

            // Greedy
            while (tauHigh - tauLow - delta >= 1e-9)
            {
                tau = tauLow + (tauHigh - tauLow) / 2;
                MHR mhr(tau);
                Greedy MyGreedy(k, mhr, FunctionClass);
                res = evaluate_optimizer(MyGreedy, dataP);
                if (get<0>(res) > fval_pre)
                {
                    fval_pre = get<0>(res);
                    tauLow = tau;
                }
                else
                {
                    tauHigh = tau;
                }
            }
            cout << "Selecting " << k << " representative points by Greedy" << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
            // outfile << "Selecting " << k << " representative points by Greedy" << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
            // cout << endl;
            // outfile << endl;
            outfile_fval << get<0>(res) << " ";
            outfile_time << get<1>(res) << " ";
            // outfile_maxn << get<3>(res) << " ";

            // GreedyAT
            // auto beta_GreedyAT = {0.1, 0.3, 0.5, 0.7, 0.9};
            // auto c1_GreedyAT = {0.6, 0.8, 1.0, 1.2, 1.4};
            // auto c2_GreedyAT = {0.2, 0.4, 0.6, 0.8, 1.0};
            auto beta_GreedyAT = {0.3};
            auto c1_GreedyAT = {1.2};
            auto c2_GreedyAT = {0.6};
            for(auto beta : beta_GreedyAT)
            {
                for(auto c1 : c1_GreedyAT)
                {
                    for(auto c2 : c2_GreedyAT)
                    {
                        if(c2 <= c1)
                        {
                            tauHigh = 1.0, tauLow = 0.0, fval_pre = 0.0;
                            while (tauHigh - tauLow - delta >= 1e-9)
                            {
                                tau = tauLow + (tauHigh - tauLow) / 2;
                                MHR mhr(tau);
                                GreedyAT MyGreedyAT(k, mhr, beta, c1, c2, FunctionClass);
                                res = evaluate_optimizer(MyGreedyAT, dataP);
                                if (get<0>(res) > fval_pre)
                                {
                                    fval_pre = get<0>(res);
                                    tauLow = tau;
                                }
                                else
                                {
                                    tauHigh = tau;
                                }
                            }
                            cout << "Selecting " << k << " representatives by GreedyAT with beta = " << beta << " and c1 = " << c1 << " and c2 = " << c2 << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                            // outfile << "Selecting " << k << " representatives by GreedyAT with beta = " << beta << " and c1 = " << c1 << " and c2 = " << c2 << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                            outfile_fval << get<0>(res) << " ";
                            outfile_time << get<1>(res) << " ";
                            // outfile_maxn << get<3>(res) << " ";
                        }
                    }
                }
            }
            // cout << endl;
            // outfile << endl;


            // Preemption
            // auto c_Preemption = {0.6, 0.8, 1.0, 1.2, 1.4};
            auto c_Preemption = {1.0};
            for(auto c: c_Preemption)
            {
                tauHigh = 1.0, tauLow = 0.0, fval_pre = 0.0;
                while (tauHigh - tauLow - delta >= 1e-9)
                {
                    tau = tauLow + (tauHigh - tauLow) / 2;
                    MHR mhr(tau);
                    Preemption MyPreemption(k, mhr, c, FunctionClass);
                    res = evaluate_optimizer(MyPreemption, dataP);
                    if (get<0>(res) > fval_pre)
                    {
                        fval_pre = get<0>(res);
                        tauLow = tau;
                    }
                    else
                    {
                        tauHigh = tau;
                    }
                }
                cout << "Selecting " << k << " representatives by Preemption with c = " << c << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                // outfile << "Selecting " << k << " representatives by Preemption with c = " << c << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                outfile_fval << get<0>(res) << " ";
                outfile_time << get<1>(res) << " ";
                // outfile_maxn << get<3>(res) << " ";
            }
            // cout << endl;
            // outfile << endl;


            // ThreeSieves
            // auto T_ThreeSieves = {50, 100, 500, 1000, 5000};
            // auto eps_ThreeSieves = {0.001, 0.005, 0.01, 0.05, 0.1};
            auto T_ThreeSieves = {1000};
            auto eps_ThreeSieves = {0.1};
            for(auto T : T_ThreeSieves)
            {
                for(auto eps : eps_ThreeSieves)
                {
                    tauHigh = 1.0, tauLow = 0.0, fval_pre = 0.0;
                    while(tauHigh - tauLow - delta >= 1e-9)
                    {
                        tau = tauLow + (tauHigh - tauLow) / 2;
                        MHR mhr(tau);
                        double singleton_m = evaluate_singleton_m(dataP, FunctionClass, mhr); // the largest function value of a singleton set
                        ThreeSieves MyThreeSieves(k, mhr, singleton_m, eps, ThreeSieves::THRESHOLD_STRATEGY::SIEVE, T, FunctionClass);
                        res = evaluate_optimizer(MyThreeSieves, dataP);
                        if(get<0>(res) > fval_pre)
                        {
                            fval_pre = get<0>(res);
                            tauLow = tau;
                        }
                        else
                        {
                            tauHigh = tau;
                        }
                    }
                    cout << "Selecting " << k << " representatives by ThreeSieves with T = " << T << " and epsilon = " << eps  << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                    // outfile << "Selecting " << k << " representatives by ThreeSieves with T = " << T << " and epsilon = " << eps  << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                    outfile_fval << get<0>(res) << " ";
                    outfile_time << get<1>(res) << " ";
                    // outfile_maxn << get<3>(res) << " ";
                }
            }
            
            
            // ThresholdOnline
            auto alpha_ThresholdOnline = {0.0};
            auto gamma_ThresholdOnline = {0.0};
            // auto alpha_ThresholdOnline = {0.0, 0.02, 0.04, 0.06, 0.08, 0.1};
            // auto gamma_ThresholdOnline = {0.0,0.25,0.5, 0.75,1.0, 1.25, 1.5};
            for (auto alpha : alpha_ThresholdOnline)
            {
                for(auto gamma : gamma_ThresholdOnline)
                {
                    tauHigh = 1.0, tauLow = 0.0, fval_pre = 0.0;
                    while(tauHigh - tauLow - delta >= 1e-9)
                    {
                        tau = tauLow + (tauHigh - tauLow) / 2;
                        MHR mhr(tau);
                        ThresholdOnline MyThresholdOnline(k, mhr, alpha, gamma, FunctionClass);
                        res = evaluate_optimizer(MyThresholdOnline, dataP);
                        if(get<0>(res) > fval_pre)
                        {
                            fval_pre = get<0>(res);
                            tauLow = tau;
                        }
                        else
                        {
                            tauHigh = tau;
                        }
                    }
                    cout << "Selecting " << k <<"->"<<get<2>(res)<< " representatives by ThresholdOnline with alpha = " << alpha << " and gamma = " << gamma << "\t fval:\t" << get<0>(res) << "\t runtime:\t" << get<1>(res) << endl;
                    // outfile <<  alpha << " " << gamma << " "<< get<0>(res) << " "<< get<1>(res) << " "<< get<2>(res)<< endl;
                    outfile_fval << get<0>(res) << " ";
                    outfile_time << get<1>(res) << " ";
                    // outfile_maxn << get<3>(res) << " ";
                }
            }
            // cout << endl;
            // outfile << endl;
            
            
            // SieveOnline
            // auto eps_SieveOnline = {0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9}; // the sampling accuracy
            auto eps_SieveOnline = {0.2};
            for (auto eps : eps_SieveOnline)
            {
                tauHigh = 1.0, tauLow = 0.0, fval_pre = 0.0;
                while (tauHigh - tauLow - delta >= 1e-9)
                {
                    tau = tauLow + (tauHigh - tauLow) / 2;
                    MHR mhr(tau);
                    double singleton_m = evaluate_singleton_m(dataP, FunctionClass, mhr); // the largest function value of a singleton set
                    SieveOnline MySieveOnline(k, mhr, eps, FunctionClass);
                    res = evaluate_optimizer(MySieveOnline, dataP);
                    if (get<0>(res) > fval_pre)
                    {
                        fval_pre = get<0>(res);
                        tauLow = tau;
                    }
                    else
                    {
                        tauHigh = tau;
                    }
                }
                cout << "Selecting " << k <<"->"<<get<2>(res)<< " representatives by SieveOnline with eps = " << eps << "\t fval:\t" << get<0>(res) << "\t\t runtime:\t" << get<1>(res) << endl;
                // outfile << eps << " " << get<0>(res) << " " << get<1>(res) << " "<< get<2>(res)<< endl;
                outfile_fval << get<0>(res) << " ";
                outfile_time << get<1>(res) << " ";
                // outfile_maxn << get<3>(res) << " ";
            }
            // cout << endl;
            // outfile << endl;

            cout << endl;
            // outfile << endl;
            outfile_fval << endl;
            outfile_time << endl; 
            // outfile.close();
            outfile_fval.close();
            outfile_time.close();
        }
    }
    return 0;
}