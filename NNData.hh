#ifndef __NNDATA_HH__
#define __NNDATA_HH__



#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sstream>

using namespace std;


class NNData{
  //This class is designed to hold a "data set" for a neural network, consisting of a set of
  //input vectors, and optionally, a set of corresponding "desired" output vectors.

public:
  NNData();
  virtual ~NNData();

  friend std::ostream& operator<<(std::ostream& argOStream, NNData* argNNData);
  friend std::ostream& operator<<(std::ostream& argOStream, NNData& argNNData){
    argOStream<<(&argNNData);
    return argOStream;
  };
  friend std::istream& operator>>(std::istream& argIStream, NNData* argNNData);
  friend std::istream& operator>>(std::istream& argIStream, NNData& argNNData){
    argIStream>>(&argNNData);
    return argIStream;
  }

  void AddData(vector<double>argInput,vector<double>argOutput=vector<double>()){
    fInput.push_back(argInput);
    fInputNormByEntry.push_back(1);
    if(fInputNorm.size()!=argInput.size())fInputNorm.resize(argInput.size(),1);
    if(argOutput.size()){
      fOutput.push_back(argOutput);
      if(fOutputNorm.size()!=argOutput.size())fOutputNorm.resize(argOutput.size(),1);
    }
  }

  double NormalizeVector(vector<double> &argV,double argNorm=1)
  {//scale all the weights so that the norm of the vector = argNorm
    double norm=0;
    for(size_t i=0;i<argV.size();i++){
      norm+=argV[i]*argV[i];
    }
    norm=(norm>0?sqrt(norm):1);
    for(size_t i=0;i<argV.size();i++){
      argV[i]*=argNorm/norm;
    }
    return norm;
  }


  void NormalizeInputEntries(double argNorm=1.0)
  {
    if(fInputNormByEntry.size()!=fInput.size())fInputNormByEntry.resize(fInput.size(),1);
    for(size_t i=0;i<fInput.size();i++){
      fInputNormByEntry[i]=NormalizeVector(fInput[i],argNorm);
    }
  }


  double VectorMean(vector<double> argV){
    double sum=0;
    size_t n=argV.size();
    if(n==0)return 0;
    for(size_t i=0;i<n;i++)sum+=argV[i];
    return sum/n;
  }

  void ApplyNormalizationToVector(vector<double> &argV, double argNorm=1.0){
    for(size_t i=0;i<argV.size();i++)argV[i]*=argNorm;
  }
  //Various ways to normalize the input and output vectors to have different means
  double NormalizeVectorToMean(vector<double> &argV, double argMean=0.5){
    double mean=VectorMean(argV);
    if(mean==0)return 1;
    double norm=argMean/mean;
    ApplyNormalizationToVector(argV,norm);
    return norm;
  }

  void ApplyNormalizationToColumn(size_t argI, vector<vector<double> > &argVV, double argNorm=1.0){
    if(argVV.size()==0 || argI>=argVV[0].size())return;
    vector<double> col=PopOutColumn(argVV,argI);
    ApplyNormalizationToVector(col,argNorm);
    InsertColumn(col,argVV,argI);
  }
  double NormalizeColumnToMean(size_t argI, vector<vector<double> > &argVV, double argMean=0.5){
    double norm=1;
    if(argVV.size()==0 || argI>=argVV[0].size())return norm;
    vector<double> col=PopOutColumn(argVV,argI);
    norm=NormalizeVectorToMean(col,argMean);
    InsertColumn(col,argVV,argI);
    return norm;
  }
  void ApplyNormalization(vector<vector<double> > &argVV, vector<double> &argN, vector<double> argNorm){
    if(argVV.size()==0 || argNorm.size()!=argVV[0].size())return;
    if(argN.size()!=argVV[0].size())argN.resize(argVV[0].size(),1);
    for(size_t i=0;i<argVV[0].size();i++){
      argN[i]=argNorm[i];
      ApplyNormalizationToColumn(i,argVV,argNorm[i]);
    }
  }
  void ApplyNormalization(vector<vector<double> > &argVV, vector<double> &argN, double argNorm=1.0){
    if(argVV.size()==0)return;
    vector<double> norm(argVV[0].size(),argNorm);
    ApplyNormalization(argVV,argN,norm);
  }

  void NormalizeToMean(vector<vector<double> > &argVV, vector<double> &argN, vector<double> argMean){
    if(argVV.size()==0 || argMean.size()!=argVV[0].size())return;
    if(argN.size()!=argVV[0].size())argN.resize(argVV[0].size(),1);
    for(size_t i=0;i<argVV[0].size();i++){
      argN[i]=NormalizeColumnToMean(i,argVV,argMean[i]);
    }
  }
  void NormalizeToMean(vector<vector<double> > &argVV, vector<double> &argN, double argMean=0.5){
    if(argVV.size()==0)return;
    vector<double> mean(argVV[0].size(),argMean);
    NormalizeToMean(argVV,argN,mean);
  }
  void ApplyNormalizationToInput(vector<double> argNorm){
    ApplyNormalization(fInput,fInputNorm,argNorm);
  }
  void ApplyNormalizationToInput(double argNorm=1.0){
    ApplyNormalization(fInput,fInputNorm,argNorm);
  }
  void ApplyNormalizationToInputColumn(size_t argI, double argNorm=1.0){
    if(fInput.size()==0 || argI>=fInput[0].size())return;
    if(fInputNorm.size()!=fInput[0].size())fInputNorm.resize(fInput[0].size());
    fInputNorm[argI]=argNorm;
    ApplyNormalizationToColumn(argI,fInput,argNorm);
  }
  void NormalizeInputToMean(vector<double> argMean){
    NormalizeToMean(fInput,fInputNorm,argMean);
  }
  void NormalizeInputToMean(double argMean=0.5){
    NormalizeToMean(fInput,fInputNorm,argMean);
  }
  void NormalizeInputColumnToMean(size_t argI, double argMean){
    if(fInput.size()==0 || argI>=fInput[0].size())return;
    if(fInputNorm.size()!=fInput[0].size())fInputNorm.resize(fInput[0].size());
    fInputNorm[argI]=NormalizeColumnToMean(argI,fInput,argMean);
  }

  void ApplyNormalizationToOutput(vector<double> argNorm){
    ApplyNormalization(fOutput,fOutputNorm,argNorm);
  }
  void ApplyNormalizationToOutput(double argNorm=1.0){
    ApplyNormalization(fOutput,fOutputNorm,argNorm);
  }
  void ApplyNormalizationToOutputColumn(size_t argI, double argNorm=1.0){
    if(fOutput.size()==0 || argI>=fOutput[0].size())return;
    if(fOutputNorm.size()!=fOutput[0].size())fOutputNorm.resize(fOutput[0].size());
    fOutputNorm[argI]=argNorm;
    ApplyNormalizationToColumn(argI,fOutput,argNorm);
  }
  void NormalizeOutputToMean(vector<double> argMean){
    NormalizeToMean(fOutput,fOutputNorm,argMean);
  }
  void NormalizeOutputToMean(double argMean=0.5){
    NormalizeToMean(fOutput,fOutputNorm,argMean);
  }
  void NormalizeOutputColumnToMean(size_t argI, double argMean){
    if(fOutput.size()==0 || argI>=fOutput[0].size())return;
    if(fOutputNorm.size()!=fOutput[0].size())fOutputNorm.resize(fOutput[0].size());
    fOutputNorm[argI]=NormalizeColumnToMean(argI,fOutput,argMean);
  }

  vector<double> PopOutColumn(vector<vector<double> > &argVV, int argI=-1){//remove a column from a set of vector (e.g. to remove one type of input or output
    size_t index=(argI==-1 ? argVV[0].size()-1:(size_t)argI);
    vector<double> pop(0);
    if(argVV.size()==0 || index>=argVV[0].size())return pop;
    for(size_t i=0;i<argVV.size();i++){
      pop.push_back(argVV[i][index]);
      argVV[i].erase(argVV[i].begin()+index);
    }
    return pop;
  }
  vector<double> PopOutInputColumn(int argI=-1){//remove an input column (last column if not specified)
    size_t index=(argI==-1 ? fInput[0].size()-1:(size_t)argI);
    if(fInputNorm.size()>=index)fInputNorm.erase(fInputNorm.begin()+index);
    return PopOutColumn(fInput,argI);
  }
  vector<double> PopOutOutputColumn(int argI=-1){//remove an output column (last column if not specified)
    size_t index=(argI==-1 ? fOutput[0].size()-1:(size_t)argI);
    if(fOutputNorm.size()>=index)fOutputNorm.erase(fOutputNorm.begin()+index);
    return PopOutColumn(fOutput,argI);
  }

  void InsertColumn( vector<double> argIV,vector<vector<double> > &argVV, int argI=-1){//(last column if not specified)
    size_t index=(argI==-1 ? argVV[0].size():(size_t)argI);
    if(argVV.size()==0 || index>argVV[0].size() || argIV.size()!=argVV.size())return;
    for(size_t i=0;i<argVV.size();i++){
      argVV[i].insert(argVV[i].begin()+index,argIV[i]);
    }
  }
  void InsertInputColumn(vector<double> argIV, double argNorm=1, int argI=-1){//(last column if not specified)
    //doesn't normalize the input column!
    size_t index=(argI==-1 ? fInput[0].size():(size_t)argI);
    if(fInputNorm.size()>0)fInputNorm.insert(fInputNorm.begin()+index,argNorm);
    InsertColumn(argIV,fInput,argI);
  }
  void InsertOutputColumn(vector<double> argIV, double argNorm=1, int argI=-1){//(last column if not specified)
    size_t index=(argI==-1 ? fOutput[0].size():(size_t)argI);
    if(fOutputNorm.size()>0)fOutputNorm.insert(fOutputNorm.begin()+index,argNorm);
    InsertColumn(argIV,fOutput,argI);
  }

  vector<double> GetColumn(size_t argI, vector<vector<double> > argVV){//returns a copy of the column
    vector<double> col(0);
    if(argVV.size()==0||argI>=argVV[0].size())return col;
    for(size_t i=0;i<argVV.size();i++){
      col.push_back(argVV[i][argI]);
    }
    return col;
  }
  vector<double> GetInputColumn(size_t argI){return GetColumn(argI,fInput);}//returns a copy (cannot modify it)
  vector<double> GetOutputColumn(size_t argI){return GetColumn(argI,fOutput);}
  size_t GetnInput(){return (fInput.size()>0 ? fInput[0].size():0);};
  size_t GetnOutput(){return (fOutput.size()>0 ? fOutput[0].size():0);};
  vector<double> GetInput(size_t argI){return fInput[argI];}
  double GetInput(size_t argI,size_t argJ){return fInput[argI][argJ];}
  vector<double> GetOutput(size_t argI){return fOutput[argI];}
  double GetOutput(size_t argI,size_t argJ){return fOutput[argI][argJ];}
  vector<double> GetInputNorm(){return fInputNorm;}
  vector<double> GetOutputNorm(){return fOutputNorm;}
  double GetInputNorm(size_t argI){return fInputNorm[argI];}
  vector<double> GetInputNormByEntry(){return fInputNormByEntry;}
  double GetInputNormByEntry(size_t argI){return fInputNormByEntry[argI];}
  double GetOutputNorm(size_t argI){return fInputNorm[argI];}

  size_t GetnData(){return fInput.size();}

private:
  size_t fVersion;
  vector<vector<double> > fInput;
  vector<double> fInputNormByEntry;//if each entry is normalized
  vector<double> fInputNorm;//value to normalize the inputs
  vector<vector<double> > fOutput;
  vector<double> fOutputNorm;
};





#endif //NNDATA not defined
