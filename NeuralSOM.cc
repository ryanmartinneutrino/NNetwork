#include "NeuralSOM.hh"

NeuralSOM::NeuralSOM():fVersion(0),fnXSOM(0),fnYSOM(0),
fLastBMUNeuron(NULL),fLastBMUSquareDistance(0)
{

}

NeuralSOM::~NeuralSOM()
{

}

std::ostream& operator<<(std::ostream& argOStream, NeuralSOM* argNeuralNet)
{
  //Write the base class first
  argOStream<<(NeuralNet*)argNeuralNet;

  if(argNeuralNet->fVersion==0){
    argOStream<<argNeuralNet->fVersion<<endl;
    argOStream<<argNeuralNet->fnXSOM<<" "<<argNeuralNet->fnYSOM<<endl;
  }
  else {cout<<"ERROR: Version for file NeuralSOM output not recognized"<<endl;}
  return argOStream;
}

std::istream& operator>>(std::istream& argIStream, NeuralSOM* argNeuralNet)
{
  //Load the base class first
  argIStream>>(NeuralNet*)argNeuralNet;

  argIStream>>argNeuralNet->fVersion;
  if(argNeuralNet->fVersion==0){
    argIStream>>argNeuralNet->fnXSOM>>argNeuralNet->fnYSOM;
  }
  else {cout<<"ERROR: Version for file NeuralSOM input not recognized"<<endl;}
  return argIStream;
}

void NeuralSOM::InitializeSelfOrganizingMap(size_t argnInputs, size_t argNx, size_t argNy, NeuralNetType argType)
{//use a hex grid
  fNeuralNetType=argType;
  fNeuron.resize(2);
  fnLayer=fNeuron.size();
  fnXSOM=argNx;
  fnYSOM=argNy;

  //initialize the input layer
  size_t layer=0;
  fNeuron[layer].resize(argnInputs);
  fnNeuronInLayer.push_back(fNeuron[layer].size());

  vector<double> position(3,0);
  position[2]=layer;
  for(size_t i=0;i<fnNeuronInLayer[layer];i++){
    position[0]=i;
    fNeuron[layer][i]=new NeuronSOM(kInputNeuron);
    fNeuron[layer][i]->SetPosition(position);
    fNeuron[layer][i]->SetOutput(1);
  }

  //Initialize the map layer
  layer=1;
  position[2]=layer;
  fNeuron[layer].resize(fnXSOM*fnYSOM);
  fnNeuronInLayer.push_back(fNeuron[layer].size());
  size_t count=0;
  double sin60=sqrt(3.0)/2.0;

  for(size_t iy=0;iy<fnYSOM;iy++){
    if(fNeuralNetType==kHexSelfOrganizingMap)position[1]=iy*sin60;
    else position[1]=iy;
    for(size_t ix=0;ix<fnXSOM;ix++){
      if(iy%2==0 || fNeuralNetType==kSquareSelfOrganizingMap)position[0]=ix;
      else position[0]=ix+0.5;
      fNeuron[layer][count]= new NeuronSOM(kPerceptron);
      fNeuron[layer][count]->SetPosition(position);
      fNeuron[layer][count]->AddInputNeurons(fNeuron[layer-1],Random::GetRandomDoubleVector(fnNeuronInLayer[layer-1],-1,1));
      fNeuron[layer][count]->NormalizeInputWeights();
      count++;
    }
  }
}

NeuronSOM* NeuralSOM::GetBMUNeuron(vector<double> argInputs, vector<Neuron*> argNeuron)
{
  double weightDistance=0;
  double minDistance=fnXSOM*fnYSOM*10;
  size_t BMUIndex=0;

  for(size_t i=0;i<argNeuron.size();i++){
    weightDistance=argNeuron[i]->GetInputWeightSquareDistanceFromVector(argInputs);
    //cout<<"neuron "<<i<<" weightDistance "<<weightDistance<<endl;
    if(weightDistance<minDistance){
      minDistance=weightDistance;
      BMUIndex=i;
     }
  }
  fLastBMUSquareDistance=minDistance;
  fLastBMUNeuron=(NeuronSOM*)argNeuron[BMUIndex];
  return fLastBMUNeuron;
}

double NeuralSOM::GetDataVariance(NNData argData)
{
  double variance=0;
  size_t nData=argData.GetnData();
  for(size_t i=0;i<nData;i++){
    variance+=GetBMUNeuronSquareDistance(argData.GetInput(i),1);
  }
  return variance/double(nData);
}

void NeuralSOM::TrainSelfOrganizingMap(NNData training, size_t argnMaxIter, double argInitialLearningRate, bool argRandomize)
{
  double nx2=fnXSOM*fnXSOM;
  double ny2=fnYSOM*fnYSOM;
  double initialRadius=sqrt(nx2+ny2);
  int tenpercent=0.1*argnMaxIter;

  double lambda=double(argnMaxIter)/log(initialRadius);
  size_t nTrainingData=training.GetnData();
  size_t iteration=0;
  double currentRadius=0;
  NeuronSOM* bmuNeuron;

  cout<<"Initial learning rate: "<<argInitialLearningRate<<", initial radius "<<initialRadius<<endl;
  for(iteration=0;iteration<argnMaxIter;iteration++){
    fLearningRate=argInitialLearningRate*exp(-double(iteration)/double(argnMaxIter));
    currentRadius=initialRadius*exp(-double(iteration)/lambda);
    if(iteration%tenpercent==0){
      cout<<"At iteration "<<iteration<<" of "<<argnMaxIter<<" iterations. learning rate:"<<fLearningRate<<", radius:"<<currentRadius<<" variance : "<<GetDataVariance(training)<<endl;
    }
    if(!argRandomize){
      for(size_t i=0;i<training.GetnData();i++){//loop over all the data
        bmuNeuron=GetBMUNeuron(training.GetInput(i),1);
        bmuNeuron->AdjustWeightsSOM(training.GetInput(i),fNeuron[1],currentRadius,fLearningRate);
      }//end loop over training sets
    }
    if(argRandomize){//choose 1 data set at random
      for(size_t count=0;count<1;count++){
        size_t i=(rand() % nTrainingData);
        //Find the BMU
        bmuNeuron=GetBMUNeuron(training.GetInput(i),1);
        bmuNeuron->AdjustWeightsSOM(training.GetInput(i),fNeuron[1],currentRadius,fLearningRate);
      }
    }
  }

}






