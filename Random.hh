#ifndef RANDOM_H
#define RANDOM_H

#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

class Random
{
  public:
    Random();
    virtual ~Random();

    static double GetRandomDoubleValue(double argMin=-1, double argMax=1, double argPrecision=0.01){
      double smallest=(fabs(argMin)>fabs(argMax)? fabs(argMax):fabs(argMin));
      double precision=(smallest<100.*argPrecision?smallest/100.:argPrecision);//so that there are at least 100 numbers between the min and max

      int range = int((argMax-argMin)/precision);
      double halfRange=0.5*(argMax-argMin)/precision;
      //cout<<argMin<<" "<<argMax<<" "<<range<<" "<<halfRange<<" "<<argPrecision<<" "<<(double(rand() % range +1)-halfRange)*argPrecision <<endl;
      return (double(rand() % range +1)-halfRange)*precision;
      }

    static vector<double> GetRandomDoubleVector(size_t argN, double argMin=-1, double argMax=1, double argPrecision=0.01){
      vector<double> temp(argN);
      for(size_t i=0;i<argN;i++)temp[i]=GetRandomDoubleValue(argMin,argMax,argPrecision);
      return temp;
    }

    static size_t GetRandomIndex(size_t argMax){//returns an int from 0 (included) to argMax (excluded)
      return rand() % argMax;
    }
  protected:
  private:

};

#endif // RANDOM_H
