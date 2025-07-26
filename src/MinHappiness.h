#ifndef MINIMUM_HAPPINESS_RATIO_H
#define MINIMUM_HAPPINESS_RATIO_H

#include <mutex>
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "ObjectiveFunction.h"
#include "Point.h"
#include "RMSUtils.h"
using namespace std;

class MHR
{
private:
protected:
    double mhr_val;
    double tau;

public:
    MHR(double tau) : mhr_val(0), tau(tau) {}

    double operator()(vector<Point> const &cur_solution, vector<UtilityFunction> const &FunctionClass)
    {
        double sum = 0;
        for (size_t j = 0; j < FunctionClass.size(); ++j)
        {
            double happy_ratio, happy_max = 0, happy_tmp;
            for (size_t i = 0; i < cur_solution.size(); ++i)
            {
                happy_tmp = FunctionClass[j].direction.dotP(cur_solution[i]);
                happy_max = happy_max > happy_tmp ? happy_max : happy_tmp;
            }
            happy_ratio = min(happy_max / FunctionClass[j].fmax, tau);
            sum += happy_ratio;
        }
        mhr_val = sum / FunctionClass.size();
        return mhr_val;
    }

    double peek(vector<Point> const &cur_solution, vector<UtilityFunction> const &FunctionClass, Point const &cur_point, bool is_streaming)
    {
        vector<Point> tmp_solution(cur_solution);
        tmp_solution.push_back(cur_point);
        double ftemp;
        ftemp = this->operator()(tmp_solution, FunctionClass);
        return ftemp;
    }

    void update(vector<Point> &cur_solution, Point const &cur_point)
    {
        cur_solution.push_back(cur_point);
    }

    double get_mhr_val()
    {
        return mhr_val;
    }

    double get_tau()
    {
        return tau;
    }

    ~MHR() {}
};

#endif