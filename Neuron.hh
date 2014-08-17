#ifndef __NEURON_HH__
#define __NEURON_HH__



#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <time.h>
//#include <random>
//#include <chrono>
#include <map>
#include <algorithm>

#include "Random.hh"

using namespace std;


enum NeuronType {kInputNeuron,kConstantNeuron,kPerceptron,kOutputNeuron};
inline std::istream & operator>>(std::istream & argIS, NeuronType & argNeuronType) {
  //for streaming an enum
  unsigned int nt = 0;
  argIS >> nt;
  argNeuronType = static_cast<NeuronType>(nt);
  return argIS;
}

enum ActivationType{kSigmoid,kLinearCombo};
inline std::istream & operator>>(std::istream & argIS, ActivationType & argActivationType) {
  //for streaming an enum
  unsigned int nt = 0;
  argIS >> nt;
  argActivationType = static_cast<ActivationType>(nt);
  return argIS;
}


class Neuron{

public:
  // Neuron();
  Neuron(NeuronType argType=kPerceptron, ActivationType argActType=kSigmoid);
  virtual ~Neuron();

  friend std::ostream& operator<<(std::ostream& argOStream, Neuron* argNeuron);
  friend std::ostream& operator<<(std::ostream& argOStream, Neuron& argNeuron){
    argOStream<<(&argNeuron);
    return argOStream;
  };
  friend std::istream& operator>>(std::istream& argIStream, Neuron* argNeuron);
  friend std::istream& operator>>(std::istream& argIStream, Neuron& argNeuron){
    argIStream>>(&argNeuron);
    return argIStream;
  }

  void PerceptronActivation();//update the fOutput using a perceptron linear combination
  void CalculateBackPropErrorGradients(double argDesiredOutput=0);
  void UpdateWeightsFromBackProp(double argLearningRate=0.5, bool argUseRPROP=false, double argEtap=1.2, double argEtam=0.5);

  double GetDistanceToNeuron(Neuron* argNeuron, size_t argDim=2){
    double sum=0;
    for(size_t i=0;i<argDim;i++)sum+=pow(argNeuron->GetPosition(i)-GetPosition(i),2);
    return sqrt(sum);
  }

  void SortMapByValue(map<Neuron*, double> &argMap, vector<double> &argSortedValues, vector<Neuron*> &argSortedKeys);

  void FillAllNeuronDistanceMap(vector<Neuron*> argNeuron, size_t argDim=2){
    fAllNeuronDistanceMap.clear();
    for(size_t i=0;i<argNeuron.size();i++){
      fAllNeuronDistanceMap[argNeuron[i]]=GetDistanceToNeuron(argNeuron[i],argDim);
    }
    SortMapByValue(fAllNeuronDistanceMap,fAllNeuronSortedDistance,fAllNeuronSorted);
   // cout<<"refilling distance map for position "<<fPosition[0]<<":"<<fPosition[1]<<endl;
  }

  double GetInputWeightSquareDistanceFromVector(vector<double> argV, bool argRelative=false){
    //if relative distance, each distance is the relative distance
    if(argV.size()!=fInputWeight.size()){
      cout<<"ERROR: cannot compare vectors of different size!"<<endl;
      return 0;
    }
    else{
      double sum=0;
      double d=0;
      for(size_t i=0;i<argV.size();i++){
        d=(argV[i]-fInputWeight[i]);
        if(argRelative && argV[i]!=0)d/=(argV[i]*argV[i]);
        else if(argRelative && fInputWeight[i]!=0)d/=(fInputWeight[i]*fInputWeight[i]);
        else{}
        sum+=d*d;
      }
      return sum;
    }
  }

  double GetInputWeightDistanceFromVector(vector<double> argV, bool argRelative=false){
    return sqrt(GetInputWeightSquareDistanceFromVector(argV,argRelative));
  }

  void SetOutput(double argO){fOutput=argO;}
  double GetOutput(){return fOutput;}

  void RandomizeWeights(){
    if(fInputWeight.size()<1)return;
    for(size_t i=0;i<fInputWeight.size();i++){
      fInputWeight[i]=Random::GetRandomDoubleValue(-1./(fnInputNeuron+1),1./(fnInputNeuron+1));
      cout<<"initial weight "<<fInputWeight[i]<<endl;
    }
  }

  void NormalizeInputWeights(double argNorm=1){
    double norm=0;
    for(size_t i=0;i<fInputWeight.size();i++){
      norm+=fInputWeight[i]*fInputWeight[i];
    }
    norm=(norm>0?sqrt(norm):1);
    for(size_t i=0;i<fInputWeight.size();i++){
      fInputWeight[i]*=argNorm/norm;
    }
  }
  //Note: adding a neuron as an input to this neuron automatically adds this neuron to the output of the neuron being added
  void AddInputNeuron(Neuron* argNeuron, double argW, double argLearningRate=0.5, double argMomentum=0.5);
  void AddInputNeuron(Neuron* argNeuron){
    AddInputNeuron(argNeuron,Random::GetRandomDoubleValue());
  }//uses a random weight
  void AddInputNeurons(vector<Neuron*> &argNeuron, vector<double> argW){
    if(argNeuron.size() != argW.size())cout<<"ERROR: weight vector not the same size as input neuron vector"<<endl;
    for(size_t i=0;i<argNeuron.size();i++)AddInputNeuron(argNeuron[i],argW[i]);
  }
  void AddInputNeurons(vector<Neuron*> &argNeuron){
    for(size_t i=0;i<argNeuron.size();i++)AddInputNeuron(argNeuron[i]);
  }

  void SetInputNeuron(size_t argI, Neuron* argNeuron){//only sets the pointers (for use in resetting the pointers, e.g. after file loading)
    if(argI>fnInputNeuron-1)cout<<"ERROR: Outside of range"<<endl;
    fInputNeuron[argI]=argNeuron;
    argNeuron->AddOutputNeuron(this);//assumes this has been reset!!!
  }

  void SetInputNeurons(vector<Neuron*> argNeuron){
    if(argNeuron.size()!=fnInputNeuron)cout<<"ERROR: Wrong size of array for setting input neuron pointers"<<endl;
    for(size_t i=0;i<fnInputNeuron;i++)SetInputNeuron(i,argNeuron[i]);
  }

  void SetErrorGradientToAverageErrorGradients(double argFactor=1.0){
    for(size_t i=0;i<fBackPropInputErrorGradient.size();i++)fBackPropInputErrorGradient[i]=argFactor*fBackPropInputAvgErrorGradient[i];
  }
  void ScaleInputAverageErrorGradients(double argFactor=1.0){
    for(size_t i=0;i<fBackPropInputAvgErrorGradient.size();i++)fBackPropInputAvgErrorGradient[i]=argFactor;
  }
  size_t GetnInputNeuron(){return fnInputNeuron;}

  vector<double> GetInputWeight(){return fInputWeight;}
  double GetInputWeight(size_t i){return (i>=fnInputNeuron? -1111:fInputWeight[i]);}
  double GetInputWeight(Neuron* argNeuron);//return the input weight of the given neuron (assumed to be an input neuron of this neuron)
  void SetInputWeight(size_t i, double argW){if(i<fnInputNeuron)fInputWeight[i]=argW;}
  void ResetBackPropInputAvgErrorGradients(){fill(fBackPropInputAvgErrorGradient.begin(),fBackPropInputAvgErrorGradient.end(),0);}
  void AddOutputNeuron(Neuron* argN){fOutputNeuron.push_back(argN);fnOutputNeuron++;}
  size_t GetnOutputNeuron(){return fnOutputNeuron;}

  double GetBackPropErrorGradient(){return fBackPropErrorGradient;};
  void SetSigmoidSlope(double argSlope){fSigmoidSlope=argSlope;}

  void UseMomentum(bool argT=true){fUseMomentum=argT;}


  vector<double> GetPosition(){return fPosition;}
  double GetPosition(size_t argI){return (argI>fPosition.size()-1 ? 0: fPosition[argI]);}
  void SetPosition(vector<double> argPos){fPosition=argPos;fnPosition=fPosition.size();}

  NeuronType GetNeuronType(){return fNeuronType;}


protected:
  vector<double> fAllNeuronSortedDistance;//distance to all neurons, sorted
  vector<Neuron* > fAllNeuronSorted;//array of all neurons for which distances will be needed (to speed up determining which neurons are near)
  map<Neuron*,double> fAllNeuronDistanceMap;//map of all neurons and their distance
  vector<Neuron*> fInputNeuron;//array of input neurons
  size_t fnInputNeuron;
  vector<double> fInputWeight;


private:
  size_t fVersion;//change this for different serialization options

  double fBackPropErrorGradient;//error gradient for use in the back propagation algorithm
  vector<double> fBackPropInputErrorGradient;
  vector<double> fBackPropInputAvgErrorGradient;//specific for each weight (=ErrorGradient*inputweight)
  vector<double> fBackPropInputPreviousAvgErrorGradient;//for use in the RPROP algorithm
  vector<double> fRPROPInputDelta;
  vector<double> fPreviousInputWeightChange;

  vector<double> fLearningRate;//each weight can have its own learning rate and momentum
  vector<double> fMomentum;
  bool fUseMomentum;

  vector<Neuron*> fOutputNeuron;//array of output neurons that have this neuron as input
  size_t fnOutputNeuron;

  double fOutput;//output value
  NeuronType fNeuronType;
  ActivationType fActivationType;
  double fSigmoidSlope;//steepness of sigmoid activation function

  vector<double> fPosition;
  size_t fnPosition;


};



#endif //__NEURON_HH__ not defined
