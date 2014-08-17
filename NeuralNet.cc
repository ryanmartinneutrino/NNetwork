#include "NeuralNet.hh"


NeuralNet::NeuralNet(NeuralNetType argType):fVersion(0),fnLayer(0),fnNeuronInLayer(0),
  fNeuron(0),fNeuralNetType(argType)
{
  srand (time(NULL));
}

NeuralNet::~NeuralNet()
{


}

std::ostream& operator<<(std::ostream& argOStream, NeuralNet* argNeuralNet)
{
  if(argNeuralNet->fVersion==0){
    argOStream<<argNeuralNet->fVersion<<" "<<argNeuralNet->fNeuralNetType<<" ";
    argOStream<<argNeuralNet->fLearningRate<<" ";
    argOStream<<argNeuralNet->fnLayer<<"\n";

    for(size_t layer=0;layer<argNeuralNet->fnLayer;layer++){
      argOStream<<argNeuralNet->fnNeuronInLayer[layer]<<"\n";
      for(size_t i=0;i<argNeuralNet->fnNeuronInLayer[layer];i++){
        argOStream<<argNeuralNet->fNeuron[layer][i];
      }
    }

  }
  else {cout<<"ERROR: Version for file NeuralNet output not recognized"<<endl;}
  return argOStream;
}
std::istream& operator>>(std::istream& argIStream, NeuralNet* argNeuralNet)
{
  argIStream>>argNeuralNet->fVersion;
  if(argNeuralNet->fVersion==0){
    argIStream>>argNeuralNet->fNeuralNetType;
    argIStream>>argNeuralNet->fLearningRate;

    argIStream>>argNeuralNet->fnLayer;
    argNeuralNet->fnNeuronInLayer.resize(argNeuralNet->fnLayer);
    argNeuralNet->fNeuron.resize(argNeuralNet->fnLayer);

    for(size_t layer=0;layer<argNeuralNet->fnLayer;layer++){
      argIStream>>argNeuralNet->fnNeuronInLayer[layer];
      argNeuralNet->fNeuron[layer].resize(argNeuralNet->fnNeuronInLayer[layer]);
      for(size_t i=0;i<argNeuralNet->fnNeuronInLayer[layer];i++){
        argNeuralNet->fNeuron[layer][i]=new Neuron();
        argIStream>>argNeuralNet->fNeuron[layer][i];
      }
    }
  }
  else {cout<<"ERROR: Version for file NeuralNet input not recognized"<<endl;}
  return argIStream;

}

void NeuralNet::PrintNeuralNet()
{
  for(size_t layer=0;layer<fnLayer;layer++){
    cout<<"Layer: "<<layer<<endl;
    for(size_t i=0;i<fnNeuronInLayer[layer];i++){
      cout<<fNeuron[layer][i]->GetOutput()<<" ";
    }
    cout<<endl<<endl;
  }
}











