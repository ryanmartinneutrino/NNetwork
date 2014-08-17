#include "Neuron.hh"


Neuron::Neuron(NeuronType argType, ActivationType argActType):fVersion(0),fInputNeuron(0),fOutputNeuron(0),
  fInputWeight(0),fnInputNeuron(0),fnOutputNeuron(0),fOutput(0),fSigmoidSlope(1.0),
  fUseMomentum(false),fLearningRate(0),fMomentum(0),
  fBackPropErrorGradient(1),fBackPropInputAvgErrorGradient(0),fBackPropInputErrorGradient(0),
  fBackPropInputPreviousAvgErrorGradient(0),fRPROPInputDelta(0),fPreviousInputWeightChange(0),
  fPosition(0),fnPosition(0),
  fAllNeuronDistanceMap(), fAllNeuronSortedDistance(0),fAllNeuronSorted(0),
  fNeuronType(argType),fActivationType(argActType)
{
}

Neuron::~Neuron()
{

}

std::ostream& operator<<(std::ostream& argOStream, Neuron* argNeuron)
{
  if(argNeuron->fVersion==0){
    argOStream<<argNeuron->fVersion<<" "<<argNeuron->fNeuronType<<" ";
    argOStream<<argNeuron->fActivationType<<" "<<argNeuron->fOutput<<" "<<argNeuron->fSigmoidSlope<<" ";
    argOStream<<argNeuron->fUseMomentum<<" ";
    argOStream<<argNeuron->fnPosition<<" ";
     for(size_t i=0;i<argNeuron->fnPosition;i++){
      argOStream<<argNeuron->fPosition[i]<<" ";
    }

    argOStream<<argNeuron->fnInputNeuron<<" ";
    for(size_t i=0;i<argNeuron->fnInputNeuron;i++){
      argOStream<<argNeuron->fInputWeight[i]<<" ";
      argOStream<<argNeuron->fLearningRate[i]<<" ";
      argOStream<<argNeuron->fMomentum[i]<<" ";
    }
    //argOStream<<argNeuron->fnOutputNeuron<<" ";

    argOStream<<"\n";
  }
  else{cout<<"ERROR Version for file output not recognized"<<endl;}
  return argOStream;
}

std::istream& operator>>(std::istream& argIStream, Neuron* argNeuron)
{
  argIStream>>argNeuron->fVersion;
  if(argNeuron->fVersion==0){
    argIStream>>argNeuron->fNeuronType>>argNeuron->fActivationType>>argNeuron->fOutput>>argNeuron->fSigmoidSlope;
    argIStream>>argNeuron->fUseMomentum;
    argIStream>>argNeuron->fnPosition;
    argNeuron->fPosition.resize(argNeuron->fnPosition);
    for(size_t i=0;i<argNeuron->fnPosition;i++){
      argIStream>>argNeuron->fPosition[i];
    }

    argIStream>>argNeuron->fnInputNeuron;
    argNeuron->fInputWeight.resize(argNeuron->fnInputNeuron);
    argNeuron->fLearningRate.resize(argNeuron->fnInputNeuron);
    argNeuron->fMomentum.resize(argNeuron->fnInputNeuron);
    for(size_t i=0;i<argNeuron->fnInputNeuron;i++){
      argIStream>>argNeuron->fInputWeight[i];
      argIStream>>argNeuron->fLearningRate[i];
      argIStream>>argNeuron->fMomentum[i];
    }
    argNeuron->fInputNeuron.resize(argNeuron->fnInputNeuron);
    argNeuron->fBackPropInputErrorGradient.resize(argNeuron->fnInputNeuron);
    argNeuron->fBackPropInputAvgErrorGradient.resize(argNeuron->fnInputNeuron,0.0);
    argNeuron->fBackPropInputPreviousAvgErrorGradient.resize(argNeuron->fnInputNeuron,0.0);
    argNeuron->fRPROPInputDelta.resize(argNeuron->fnInputNeuron,0.1);
    argNeuron->fPreviousInputWeightChange.resize(argNeuron->fnInputNeuron,0.1);
  }
  else{cout<<"ERROR Version for file input not recognized"<<endl;}
  return argIStream;
}


void Neuron::AddInputNeuron(Neuron* argNeuron, double argW, double argLearningRate, double argMomentum)
{
  fInputNeuron.push_back(argNeuron);
  fInputWeight.push_back(argW);
  fLearningRate.push_back(argLearningRate);
  fMomentum.push_back(argMomentum);
  fBackPropInputErrorGradient.push_back(0.0);
  fBackPropInputAvgErrorGradient.push_back(0.0);
  fBackPropInputPreviousAvgErrorGradient.push_back(0.0);
  fRPROPInputDelta.push_back(0.1);
  fPreviousInputWeightChange.push_back(0.1);
  fnInputNeuron++;
  argNeuron->AddOutputNeuron(this);
}


void Neuron::PerceptronActivation()
{
  if(fNeuronType==kConstantNeuron || fNeuronType==kInputNeuron)return;
  //Calculate the linear combination first:
  double sum=0;
  for(size_t i=0;i<fnInputNeuron;i++){
    //cout<<fInputWeight[i]<<" : "<<fInputNeuron[i]->GetOutput()<<endl;
    sum+=fInputWeight[i]*fInputNeuron[i]->GetOutput();
  }
  //Apply the activation function
  if(fActivationType==kSigmoid){
    if(sum>700){
      //cout<<"WARNING: inputs to this neuron are very big, maybe something is not normalized? There might be a seg fault coming up..."<<endl;
    }
    sum=1./(1+exp(-fSigmoidSlope*sum));
  }
  if(fActivationType==kLinearCombo){
    //nothing, just return the sum anyway
  }

  fOutput=sum;
}


double Neuron::GetInputWeight(Neuron* argNeuron)
{
  double val=-1111;
  for(size_t i=0;i<fnInputNeuron;i++){
    if(argNeuron==fInputNeuron[i])val=fInputWeight[i];
  }
  return val;

}

void Neuron::CalculateBackPropErrorGradients(double argDesiredOutput)
{
  if(fNeuronType==kInputNeuron || fNeuronType==kConstantNeuron)return;

  if(fNeuronType==kOutputNeuron){//Calculate the error gradients for an output neuron
    if(fActivationType==kSigmoid){
      fBackPropErrorGradient=(fOutput-argDesiredOutput)*fSigmoidSlope*fOutput*(1.-fOutput);
    }
    else if(fActivationType==kLinearCombo){
      fBackPropErrorGradient=(fOutput-argDesiredOutput);
    }
    else fBackPropErrorGradient=0;
    for(size_t i=0;i<fnInputNeuron;i++){
      fBackPropInputErrorGradient[i]=fInputNeuron[i]->GetOutput()*fBackPropErrorGradient;
      fBackPropInputAvgErrorGradient[i]+=fBackPropInputErrorGradient[i];
    }
    return;
  }

  if(fNeuronType==kPerceptron){//Error gradients for a hidden layer neuron
    double sum=0;
    if(fActivationType==kSigmoid){
      for(size_t k=0;k<fnOutputNeuron;k++){
        sum+=fOutputNeuron[k]->GetInputWeight(this)*fOutputNeuron[k]->GetBackPropErrorGradient();
      }
      fBackPropErrorGradient=fSigmoidSlope*fOutput*(1.0-fOutput)*sum;
    }
    else if(fActivationType==kLinearCombo){
      for(size_t k=0;k<fnOutputNeuron;k++){
        sum+=fOutputNeuron[k]->GetInputWeight(this)*fOutputNeuron[k]->GetBackPropErrorGradient();
      }
      fBackPropErrorGradient=sum;
    }
    else fBackPropErrorGradient=0;

    for(size_t i=0;i<fnInputNeuron;i++){
      fBackPropInputErrorGradient[i]=fInputNeuron[i]->GetOutput()*fBackPropErrorGradient;
      fBackPropInputAvgErrorGradient[i]+=fBackPropInputErrorGradient[i];
    }
  }
}

void Neuron::UpdateWeightsFromBackProp(double argLearningRate, bool argUseRPROP, double argEtap, double argEtam)
{
  if(fNeuronType==kInputNeuron || fNeuronType==kConstantNeuron)return;

  if(!argUseRPROP){
    for(size_t i=0;i<fnInputNeuron;i++){
        fLearningRate[i]=argLearningRate;
        double weightChange=-fLearningRate[i]*fBackPropInputErrorGradient[i];
        if(fUseMomentum)weightChange+=fMomentum[i]*fPreviousInputWeightChange[i];
        fInputWeight[i]+=weightChange;
        fPreviousInputWeightChange[i]=weightChange;
        //if(fabs(fInputWeight[i])>5)fInputWeight[i]=GetRandomDoubleValue();
    }
  }

  if(argUseRPROP){
    double delta=1;
    double maxDelta=50.0;
    double minDelta=0.0000001;
    for(size_t i=0;i<fnInputNeuron;i++){
      if(fBackPropInputAvgErrorGradient[i]*fBackPropInputPreviousAvgErrorGradient[i]>0){
        fRPROPInputDelta[i]=(argEtap*fRPROPInputDelta[i]>maxDelta? maxDelta:argEtap*fRPROPInputDelta[i]);
        delta=-(fBackPropInputAvgErrorGradient[i]>=0 ? 1.0: -1.0)*fRPROPInputDelta[i];
        if(fBackPropInputAvgErrorGradient[i]==0)delta=0;
      }
      else if(fBackPropInputAvgErrorGradient[i]*fBackPropInputPreviousAvgErrorGradient[i]<0){
        fRPROPInputDelta[i]=(argEtam*fRPROPInputDelta[i]<minDelta? minDelta:argEtam*fRPROPInputDelta[i]);
        delta=-fPreviousInputWeightChange[i];
        fBackPropInputAvgErrorGradient[i]=0;
      }
      else if(fBackPropInputAvgErrorGradient[i]*fBackPropInputPreviousAvgErrorGradient[i]==0 ){
        delta=-(fBackPropInputAvgErrorGradient[i]>=0 ? 1.0: -1.0)*fRPROPInputDelta[i];
        if(fBackPropInputAvgErrorGradient[i]==0)delta=0;
      }
      fInputWeight[i]+=delta;
      //if(fabs(fInputWeight[i])>5)fInputWeight[i]=GetRandomDoubleValue();
      fBackPropInputPreviousAvgErrorGradient[i]=fBackPropInputAvgErrorGradient[i];
      fPreviousInputWeightChange[i]=delta;
    }
  }

}


void Neuron::SortMapByValue(map<Neuron*, double> &argMap,vector<double>& argSortedValues,vector<Neuron*> &argSortedKeys)
 {
   //It is impossible to sort a map, maps are automatically sorted by key!!!
   //However, this returns two vectors of the sorted values!

   argSortedKeys.clear();
   argSortedValues.clear();

   map<double, Neuron*> swappedMap;
   map<Neuron*,double>::iterator it;
   map<double, Neuron*>::iterator its;

   //create an inverse map and sort it (elements of the original map that had the same value will be lost!!)
   for(it=argMap.begin();it!=argMap.end();++it){
    swappedMap[it->second]=it->first;
   }
  //The swapped map is by defintion sorted by its keys (c++ does this for you!)

  //Fill in sorted vectors
   for(its=swappedMap.begin();its!=swappedMap.end();++its){
      //cout<<"Sorted: "<<its->first<<endl;
      for(it=argMap.begin();it!=argMap.end();++it){
        if(its->first==it->second){//all entries in the original map that have the same value are now added
          argSortedKeys.push_back(it->first);
          argSortedValues.push_back(it->second);
        }
      }
   }

  if(argSortedKeys.size()!=argMap.size())cout<<"ERROR sorting map, something went wrong..."<<endl;

 }

