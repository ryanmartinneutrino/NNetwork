#ifndef NEURALSOM_H
#define NEURALSOM_H

#include "NeuralNet.hh"
#include "NeuronSOM.hh"

class NeuralSOM : public NeuralNet
{
  public:
    NeuralSOM();
    virtual ~NeuralSOM();

    friend std::ostream& operator<<(std::ostream& argOStream, NeuralSOM* argNeuralNet);
    friend std::ostream& operator<<(std::ostream& argOStream, NeuralSOM& argNeuralNet){
      argOStream<<(&argNeuralNet);
      return argOStream;
    }
    friend std::istream& operator>>(std::istream& argIStream, NeuralSOM* argNeuralNet);
    friend std::istream& operator>>(std::istream& argIStream, NeuralSOM& argNeuralNet){
      argIStream>>(&argNeuralNet);
      return argIStream;
    }

  void InitializeSelfOrganizingMap(size_t argnInputs, size_t argNx, size_t argNy, NeuralNetType argType=kSquareSelfOrganizingMap);
  void TrainSelfOrganizingMap(NNData training, size_t argnMaxIter=10000, double argInitialLearningRate=0.1, bool argRandomize=false);

  double GetDataVariance(NNData argData);

  NeuronSOM* GetBMUNeuron(vector<double> argInputs, vector<Neuron*> argNeuron);//fills in last BMU square distance and neuron
  NeuronSOM* GetBMUNeuron(vector<double> argInputs, size_t layer=1){
    return GetBMUNeuron(argInputs,fNeuron[layer]);
  };
  double GetBMUNeuronSquareDistance(vector<double> argInputs, vector<Neuron*> argNeuron){
    GetBMUNeuron(argInputs,argNeuron);
    return fLastBMUSquareDistance;
  }
  double GetBMUNeuronSquareDistance(vector<double> argInputs, size_t layer=1){
    return GetBMUNeuronSquareDistance(argInputs,fNeuron[layer]);
  }


  protected:

  private:
    size_t fVersion;
    size_t fnXSOM;
    size_t fnYSOM;
    double fLastBMUSquareDistance;
    NeuronSOM* fLastBMUNeuron;

};

#endif // NEURALSOM_H
