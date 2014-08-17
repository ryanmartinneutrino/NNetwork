
#ifndef __NEURALNET_HH__
#define __NEURALNET_HH__

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
//#include <random>

using namespace std;

#include "Random.hh"
#include "Neuron.hh"
#include "NNData.hh"

enum NeuralNetType {kFeedForwardBackProp,kSquareSelfOrganizingMap,kFeedForwardRPROP,kHexSelfOrganizingMap};

inline std::istream & operator>>(std::istream & argIS, NeuralNetType & argNeuralNetType) {
  //for streaming an enum
  unsigned int nt = 0;
  argIS >> nt;
  argNeuralNetType = static_cast<NeuralNetType>(nt);
  return argIS;
}


class NeuralNet{

public:
  NeuralNet(NeuralNetType argType=kFeedForwardBackProp);
  virtual ~NeuralNet();

  friend std::ostream& operator<<(std::ostream& argOStream, NeuralNet* argNeuralNet);
  friend std::ostream& operator<<(std::ostream& argOStream, NeuralNet& argNeuralNet){
    argOStream<<(&argNeuralNet);
    return argOStream;
  };
  friend std::istream& operator>>(std::istream& argIStream, NeuralNet* argNeuralNet);
  friend std::istream& operator>>(std::istream& argIStream, NeuralNet& argNeuralNet){
    argIStream>>(&argNeuralNet);
    return argIStream;
  }



  void PrintNeuralNet();

  Neuron* GetRandomNeuronInLayer(size_t argLayer){return fNeuron[argLayer][Random::GetRandomIndex(fnNeuronInLayer[argLayer])];}

  NeuralNetType GetNeuralNetType(){return fNeuralNetType;}


  double GetOutput(size_t argI){return fNeuron[fnLayer-1][argI]->GetOutput();}
  size_t GetnOutput(){return fNeuron[fnLayer-1].size();}

  Neuron* GetNeuron(size_t argLayer, size_t argI){return fNeuron[argLayer][argI];}

  size_t GetnNeuron(size_t argLayer){return fnNeuronInLayer[argLayer];}
  size_t GetnActiveNeuron(size_t argLayer){//Get the number of non constant neurons in a layer
    size_t n=0;
    for(size_t i=0;i<fnNeuronInLayer[argLayer];i++){
      if(fNeuron[0][i]->GetNeuronType()!=kConstantNeuron)n++;
    }
    return n;
  }

  size_t GetnInput(){return GetnActiveNeuron(0);}

  void SetInput(vector<double> argIn){//assumes constant neurons at the end of the layer!!!
    size_t ninput=GetnInput();
    if(argIn.size()!=ninput){//the size is 1 bigger than nInputs, because of the bias neuron
      cout<<"ERROR: Input not the right size"<<endl;
      return;
    }
    for(size_t i=0;i<ninput;i++){
      fNeuron[0][i]->SetOutput(argIn[i]);
    }
  }

protected:

  NeuralNetType fNeuralNetType;
  size_t fnLayer;
  vector<size_t> fnNeuronInLayer;
  vector<vector<Neuron*> > fNeuron;
  double fLearningRate;//0.5 by default

private:

  size_t fVersion;


};


#endif //__NEURALNET_HH__ not defined
