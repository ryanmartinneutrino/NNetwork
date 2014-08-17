#include "NeuronSOM.hh"

NeuronSOM::NeuronSOM(NeuronType argType) :Neuron(argType)
{

}

NeuronSOM::~NeuronSOM()
{

}

void NeuronSOM::AdjustWeightsSOM(vector<double> argInput, vector<Neuron*> argNeuron, double argRadius, double argLearningRate)
{

  double distanceFactor=0.;
  double delta=0;
  double currentWeight;
  double distance=0;
  //Faster?

  if(fAllNeuronSortedDistance.size() != argNeuron.size())FillAllNeuronDistanceMap(argNeuron,2);

  for(size_t i=0;i<fAllNeuronSortedDistance.size();i++){
    distance=fAllNeuronSortedDistance[i];
    //if(distance>argRadius)break;
    distanceFactor=exp(-1.0* (pow(distance,2))/(2.0*pow(argRadius,2)));
    for(size_t j=0;j<fnInputNeuron;j++){
      currentWeight=fAllNeuronSorted[i]->GetInputWeight(j);
      delta=argInput[j]-currentWeight;
      fAllNeuronSorted[i]->SetInputWeight(j,currentWeight+argLearningRate*distanceFactor*delta);
    }
  }

  //adjust this neurons's weight:
  for(size_t j=0;j<fnInputNeuron;j++){
    delta=argInput[j]-fInputWeight[j];
    fInputWeight[j]+=argLearningRate*delta;
  }
}
