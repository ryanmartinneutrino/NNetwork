#ifndef NEURALMLP_H
#define NEURALMLP_H

#include "NeuralNet.hh"


class NeuralMLP : public NeuralNet
{
  public:
    NeuralMLP();
    virtual ~NeuralMLP();

    friend std::ostream& operator<<(std::ostream& argOStream, NeuralMLP* argNeuralNet);
    friend std::ostream& operator<<(std::ostream& argOStream, NeuralMLP& argNeuralNet){
      argOStream<<(&argNeuralNet);
      return argOStream;
    }
    friend std::istream& operator>>(std::istream& argIStream, NeuralMLP* argNeuralNet);
    friend std::istream& operator>>(std::istream& argIStream, NeuralMLP& argNeuralNet){
      argIStream>>(&argNeuralNet);
      return argIStream;
    }

  void InitializeFeedForwardBackPropNetwork(size_t argnInputs, size_t argnOutputs, vector<size_t> argnNeuronPerHiddenLayer, NeuralNetType argType=kFeedForwardBackProp );
  void InitializeFeedForwardBackPropNetwork(size_t argnInputs, size_t argnOutputs, size_t argnHiddenNeurons,NeuralNetType argType=kFeedForwardBackProp ){//to use a single hidden layer
    vector<size_t> nNeurons(1,argnHiddenNeurons);
    InitializeFeedForwardBackPropNetwork(argnInputs,argnOutputs, nNeurons,argType);
  }


  void RunFeedForward();
  void RunFeedForward(vector<double> argInputs){
    SetInput(argInputs);
    RunFeedForward();
  }
  void BackPropagate(vector<double> argDesired, bool argUpdateWeights=true);
  void RunTrainingDataFeedForwardBackProp(NNData argTraining);
  void RunTestingDataFeedForward(NNData argTesting, double argTolerance=0.01);//checks TPR and FPR (true if output is bigger than 1-argTolerance)
  bool TrainNeuralNetFeedForwardBackProp(NNData argTrainig, NNData argTesting,
    double argFailureRateTolerance=0.01, long unsigned int argMaxCount=100000,
    bool argAdaptiveLearningRate=false);

//Getters and setters:
  void RandomizeNeuronsWhenStuck(bool argT=true){fRandomizeNeuronsWhenStuck=argT;}
  void SetLearningRate(double argRate){fLearningRate=argRate;}
  void SetBatchUpdate(bool argT=true){fBatchUpdate=argT;}
  void SetNeuralNetType(NeuralNetType argType){fNeuralNetType=argType;}
  void SetnRandomsPerSet(size_t argnR=0){fnRandomsPerSet=argnR;}//if 0, just uses number of trsining data sets

  void AddNeuronsWhenStuck(bool argT=true){fAddNeuronsWhenStuck=argT;}

  void UseMomentum(double argM=0.5, bool argT=true){fMomentum=argM;fUseMomentum=argT;}
  //vector<double> GetTestResultVector(){return fTestResultVector;}
  //double GetTestResult(size_t argI){return fTestResultVector[argI];}
  //double GetTestResult(){return fTestResult;}
  size_t GetnConvergence(){return fnConvergence;}
  double GetTPR(){return fTPR;}
  double GetFPR(){return fFPR;}
  double GetTNR(){return fTNR;}
  double GetFNR(){return fFNR;}

  protected:


  private:
    size_t fVersion;
    vector<double> fErrorVector;
    double fErrorVectorNorm;

    long unsigned int fnConvergence;
    bool fBatchUpdate;//upload only after after looping through a whole training set;
    bool fRandomizeNeuronsWhenStuck;
    bool fAddNeuronsWhenStuck;//add neurons to the net when stuck
    size_t fnRandomsPerSet;//when randomly choosing a training set, how many to choose

    double fMomentum;
    bool fUseMomentum;

    double fTPR;//true positive rate
    double fFPR;//false positive rate
    double fTNR;//true negative rate
    double fFNR;//false negative rate

};

#endif // NEURALMLP_H
