#ifndef NEURONSOM_HH
#define NEURONSOM_HH

#include "Neuron.hh"


class NeuronSOM : public Neuron
{
  public:
    NeuronSOM(NeuronType argType=kPerceptron);
    virtual ~NeuronSOM();

    void AdjustWeightsSOM(vector<double> argInput, vector<Neuron*> argNeuron, double argRadius, double argLearningRate);

  protected:
  private:
};

#endif // NEURONSOM_HH
