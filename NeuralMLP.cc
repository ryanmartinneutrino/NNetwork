#include "NeuralMLP.hh"

NeuralMLP::NeuralMLP(): fVersion(0),
  //fTestResultVector(0),fTestResult(0),fnConvergence(0),
  fBatchUpdate(false),fnRandomsPerSet(0),fRandomizeNeuronsWhenStuck(false),fAddNeuronsWhenStuck(false),
  fUseMomentum(false),fMomentum(0.5),
  fErrorVector(0),fErrorVectorNorm(0.0),
  fTPR(0), fFPR(0),fTNR(0),fFNR(0)
{

}

NeuralMLP::~NeuralMLP()
{

}

std::ostream& operator<<(std::ostream& argOStream, NeuralMLP* argNeuralNet)
{
  //Write the base class first
  argOStream<<(NeuralNet*)argNeuralNet;

  if(argNeuralNet->fVersion==0){
    argOStream<<argNeuralNet->fVersion<<endl;
    argOStream<<argNeuralNet->fUseMomentum<<" "<<argNeuralNet->fMomentum<<endl;
    argOStream<<argNeuralNet->fTPR<<" "<<argNeuralNet->fFPR<<" "<<argNeuralNet->fTNR<<" "<<argNeuralNet->fFNR<<endl;
    /*
    argOStream<<argNeuralNet->fTestResult<<endl;
    for(size_t i=0;i<argNeuralNet->fnNeuronInLayer[argNeuralNet->fnLayer-1];i++){
      argOStream<<argNeuralNet->fTestResultVector[i]<<" ";
    }
    argOStream<<"\n";*/
  }
  else {cout<<"ERROR: Version for file NeuralSOM output not recognized"<<endl;}
  return argOStream;
}

std::istream& operator>>(std::istream& argIStream, NeuralMLP* argNeuralNet)
{
  //Load the base class first
  argIStream>>(NeuralNet*)argNeuralNet;

  argIStream>>argNeuralNet->fVersion;
  if(argNeuralNet->fVersion==0){
    argIStream>>argNeuralNet->fTPR>>argNeuralNet->fFPR>>argNeuralNet->fTNR>>argNeuralNet->fFNR;
    argIStream>>argNeuralNet->fUseMomentum>>argNeuralNet->fMomentum;

    argNeuralNet->fErrorVector.resize(argNeuralNet->fnNeuronInLayer[argNeuralNet->fnLayer-1]);
    /*
    argIStream>>argNeuralNet->fTestResult;
    for(size_t i=0;i<argNeuralNet->fnNeuronInLayer[argNeuralNet->fnLayer-1];i++){
        argIStream>>argNeuralNet->fTestResultVector[i];
    }
     //Set the pointers between neurons, depending on type of network
     */

    for(size_t layer=1;layer<argNeuralNet->fnLayer;layer++){
        for(size_t i=0;i<argNeuralNet->fnNeuronInLayer[layer];i++){
          if(argNeuralNet->fNeuron[layer][i]->GetNeuronType()==kConstantNeuron)continue;//no input
          if(argNeuralNet->fNeuron[layer][i]->GetNeuronType()==kInputNeuron)continue;//no input
          argNeuralNet->fNeuron[layer][i]->SetInputNeurons(argNeuralNet->fNeuron[layer-1]);//this also sets the output pointers AND uses push_back on that array (assumed 0 from the constructor)
        }
    }
  }
  else {cout<<"ERROR: Version for file input not recognized"<<endl;}
  return argIStream;
}




void NeuralMLP::InitializeFeedForwardBackPropNetwork(size_t argnInputs, size_t argnOutputs,vector<size_t> argnNeuronPerHiddenLayer, NeuralNetType argType)
{
  //Initialize a standard feed forward network with the given number of inputs, outputs, hidden layers and the number of neurons in each hidden layer (given by the vector argnNeuronPerHiddenLayer). Function is overloaded if only 1 hidden layer, then only the number of neuron in that hidden layer is needed.
  //A bias neuron is added in all the layers except output
  //The neurons are organized on a 2D grid with the layer being the first coordinate and the position within that layer being the second coordinate.
  if(argType!=kFeedForwardBackProp &&argType!=kFeedForwardRPROP){
    cout<<"ERROR, wrong type of neural net type for Initializing a feed forward net";
  }
  fNeuralNetType=argType;
  if(argType==kFeedForwardRPROP){
    fBatchUpdate=true;
  }

  size_t nHiddenLayer=argnNeuronPerHiddenLayer.size();
  fNeuron.resize(2+nHiddenLayer);

  //Input neuron layer:
  size_t layer=0;
  vector<double> position(2);
  position[0]=layer;
  fNeuron[layer].resize(argnInputs+1);//+1 for the bias neuron
  fnNeuronInLayer.push_back(fNeuron[layer].size());
  for(size_t i=0;i<fnNeuronInLayer[layer]-1;i++){
    fNeuron[layer][i]=new Neuron(kInputNeuron);
    fNeuron[layer][i]->SetOutput(1);
    position[1]=i;
    fNeuron[layer][i]->SetPosition(position);
  }
  fNeuron[layer][fnNeuronInLayer[layer]-1]=new Neuron(kConstantNeuron);
  fNeuron[layer][fnNeuronInLayer[layer]-1]->SetOutput(-1);
  position[1]=fnNeuronInLayer[layer]-1;
  fNeuron[layer][fnNeuronInLayer[layer]-1]->SetPosition(position);

  //Hidden layers
  for(layer=1;layer<nHiddenLayer+1;layer++){
    position[0]=layer;
    fNeuron[layer].resize(argnNeuronPerHiddenLayer[layer-1]+1);//+1 for the bias neuron
    fnNeuronInLayer.push_back(fNeuron[layer].size());
    for(size_t i=0;i<fnNeuronInLayer[layer]-1;i++){
      fNeuron[layer][i]=new Neuron();//kPerceptron by default
      fNeuron[layer][i]->AddInputNeurons(fNeuron[layer-1]);//previous layer as inputs
      position[1]=i;
      fNeuron[layer][i]->SetPosition(position);
    }
    fNeuron[layer][fnNeuronInLayer[layer]-1]=new Neuron(kConstantNeuron);
    fNeuron[layer][fnNeuronInLayer[layer]-1]->SetOutput(-1);
    position[1]=fnNeuronInLayer[layer]-1;
    fNeuron[layer][fnNeuronInLayer[layer]-1]->SetPosition(position);
  }

  //Output layer
  position[0]=layer;//note that layer got incremented at the end of the for loop
  fNeuron[layer].resize(argnOutputs);
  fErrorVector.resize(argnOutputs,0);
  fnNeuronInLayer.push_back(fNeuron[layer].size());

  for(size_t i=0;i<fnNeuronInLayer[layer];i++){
    fNeuron[layer][i]=new Neuron(kOutputNeuron);
    fNeuron[layer][i]->AddInputNeurons(fNeuron[layer-1]);//previous layer as inputs
    position[1]=i;
    fNeuron[layer][i]->SetPosition(position);
  }

  fnLayer=fnNeuronInLayer.size();

  //Randomize all the weights:
  for(size_t layer=0;layer<fnLayer;layer++){
      for(size_t i=0;i<fnNeuronInLayer[layer];i++){
        fNeuron[layer][i]->RandomizeWeights();//<< SOMETHING WRONG HERE!!!!
      }
  }
}

void NeuralMLP::RunFeedForward()
{
  for(size_t layer=1;layer<fnLayer;layer++){
    for(size_t i=0;i<fnNeuronInLayer[layer];i++){
      fNeuron[layer][i]->PerceptronActivation();
    }
  }
}

void NeuralMLP::BackPropagate(vector<double> argDesired, bool argUpdateWeights)
{
  if(argDesired.size()!=fnNeuronInLayer[fnLayer-1]){
    cout<<"ERROR in desired vector size"<<endl;
    return;
  }
  for(size_t layer=fnLayer-1;layer>0;layer--){
    for(size_t i=0;i<fnNeuronInLayer[layer];i++){
      if(fUseMomentum)fNeuron[layer][i]->UseMomentum(true);
      if(layer==fnLayer-1)fNeuron[layer][i]->CalculateBackPropErrorGradients(argDesired[i]);//for the output layer, need desired output
      else fNeuron[layer][i]->CalculateBackPropErrorGradients();
      if(argUpdateWeights){
        fNeuron[layer][i]->UpdateWeightsFromBackProp(fLearningRate);
      }
    }
  }

}


void NeuralMLP::RunTrainingDataFeedForwardBackProp(NNData argTraining)
{//Run the feedforward and back prop on a training data set
  if(fNeuralNetType==kFeedForwardRPROP)fBatchUpdate=true;

  size_t nTrainingData=argTraining.GetnData();
  size_t nData=nTrainingData;
  vector<size_t> indices(nData);
  for(size_t i=0;i<nData;i++)indices[i]=i;
  size_t index=0;

  for(size_t i=0;i<nData;i++){
    index=Random::GetRandomIndex(nData-i);//randomizes the order of looping through the dataset
    RunFeedForward(argTraining.GetInput(indices[index]));
    BackPropagate(argTraining.GetOutput(indices[index]),!fBatchUpdate);
    indices.erase(indices.begin()+index);
  }

  //Batch weight update (after accumulating error gradient over the whole data set
  if(fBatchUpdate){
    for(size_t layer=fnLayer-1;layer>0;layer--){
      for(size_t i=0;i<fnNeuronInLayer[layer];i++){
        if(fNeuralNetType==kFeedForwardRPROP){
          fNeuron[layer][i]->UpdateWeightsFromBackProp(fLearningRate,true);
          fNeuron[layer][i]->ResetBackPropInputAvgErrorGradients();
        }
        if(fNeuralNetType==kFeedForwardBackProp){
          fNeuron[layer][i]->SetErrorGradientToAverageErrorGradients(1.0/nData);//?neceaaary?
          fNeuron[layer][i]->UpdateWeightsFromBackProp(fLearningRate);
          fNeuron[layer][i]->ResetBackPropInputAvgErrorGradients();
        }
      }
    }
  }

}

void NeuralMLP::RunTestingDataFeedForward(NNData argTesting, double argTolerance)
{//Run the feedforward and back prop on a training data set
  //fill(fTestResultVector.begin(), fTestResultVector.end(), 0.);
  //fTestResult=0;
  int TP=0;//true positives
  int FP=0;//false positiive
  int TN=0;//true negatives
  int FN=0;//false negative
  int P=0;//positives
  int N=0;//negatives

  fErrorVector.resize(fnNeuronInLayer[fnLayer-1],0);

  for(size_t i=0;i<argTesting.GetnData();i++){
    RunFeedForward(argTesting.GetInput(i));
    for(size_t j=0;j<GetnOutput();j++){
        double diff=fabs(GetOutput(j)-argTesting.GetOutput(i,j));
        fErrorVector[j]+=diff;
        //fTestResult+=diff;
        if(GetOutput(j)>1.0-argTolerance){ //positive output
            if(argTesting.GetOutput(i,j)>1.0-argTolerance){//true positive
                TP++;
                P++;}
            else{//false positive
                FP++;
                N++;}
        }
        else{//negative input
            if(argTesting.GetOutput(i,j)>1.0-argTolerance){//false negative
                FN++;
                P++;}
            else{//true negative
                TN++;
                N++;}
        }
    }
  }

  fTPR=double(TP)/double(P);
  fFPR=double(FP)/double(N);
  fTNR=double(TN)/double(N);
  fFNR=double(FN)/double(P);
  fErrorVectorNorm=0;
  for(size_t j=0;j<GetnOutput();j++){
      fErrorVector[j]/=argTesting.GetnData();
      fErrorVectorNorm+=fErrorVector[j]*fErrorVector[j];
  }
  fErrorVectorNorm=sqrt(fErrorVectorNorm);
  //fTestResult/=(argTesting.GetnData()*GetnOutput());
  //fTestResult=1.- double(TP+TN)/double(P+N);//1-accuracy from http://en.wikipedia.org/wiki/Receiver_operating_characteristic
}

bool NeuralMLP::TrainNeuralNetFeedForwardBackProp(NNData argTraining, NNData argTesting,
    double argFailureRateTolerance, long unsigned int argMaxCount,
    bool argAdaptiveLearningRate)
{
  double startLearningRate=fLearningRate;
  fnConvergence=0;
  bool converged=false;
  int tenpercent=argMaxCount/10.;


  size_t nReset=0;

  double FailureRateBeforeAddingNeuron=1;
  double worstFailureRate=1;
  double previousFailureRate=10;
  double changeInFailureRateTolerance=argFailureRateTolerance/1000.;//this will trigger randomization of the neuron input weights

  do{
    if(fnConvergence%tenpercent==0)cout<<"At iteration "<<fnConvergence<<" of "<<argMaxCount<<" iterations, learning rate: "<<fLearningRate<<", TPR, FPR = "<<fTPR<<", "<<fFPR<<" error vector norm "<<fErrorVectorNorm<<endl;

    RunTrainingDataFeedForwardBackProp(argTraining);
    if(argTesting.GetnData()>0)RunTestingDataFeedForward(argTesting);
    else RunTestingDataFeedForward(argTraining);
    converged=true;
    worstFailureRate=((1-fTPR)>=fFPR?(1-fTPR):fFPR);//could add the negative rates, but probably redundant
    //cout<<fnConvergence<<":"<<worstFailureRate<<endl;
    fnConvergence++;
    nReset++;

    if(worstFailureRate>argFailureRateTolerance)converged=false;
    else break;

    if(argAdaptiveLearningRate)fLearningRate=startLearningRate*exp(-double(fnConvergence)/double(argMaxCount));

    if(fabs(worstFailureRate-previousFailureRate)<changeInFailureRateTolerance){//failure rate didn't improve significantly
      if(fRandomizeNeuronsWhenStuck && nReset>argMaxCount/20){//<-- make the 20 a variable!
        nReset=0;
        //network could be stuck in a local minimum, try to kick it out
        cout<<"Network stuck, might randomize some weights, TPR, FPR = "<<fTPR<<","<<fFPR<<" previous "<<previousFailureRate<<endl;
        for(size_t layer=1;layer<fnLayer;layer++){
          double factor=exp(-2.0*(1.0-worstFailureRate));//
          size_t nRandomNeurons=Random::GetRandomIndex(fnNeuronInLayer[layer]+1)*factor;//randomize less neurons as fTestResult approaches 0
          if(nRandomNeurons>0)cout<<"Randomizing "<<nRandomNeurons<<" in layer "<<layer<<endl;
          for(size_t i=0;i<nRandomNeurons;i++){
            GetRandomNeuronInLayer(layer)->RandomizeWeights();
          }
        }
      }//end if randomize some neurons
      else if (fAddNeuronsWhenStuck&& nReset>100){
        nReset=0;
        if(worstFailureRate<FailureRateBeforeAddingNeuron-changeInFailureRateTolerance){//adding a neuron helped
          FailureRateBeforeAddingNeuron=worstFailureRate;
          cout<<"Neural Net may be stuck, adding a neuron to the first hidden layer"<<endl;
          fNeuron[1].push_back(new Neuron());
          fnNeuronInLayer[1]++;
          fNeuron[1][fnNeuronInLayer[1]-1]->AddInputNeurons(fNeuron[0]);
          vector<double> position(2,0);
          position[1]=fnNeuronInLayer[1]-1;
          fNeuron[1][fnNeuronInLayer[1]-1]->SetPosition(position);
          cout<<"First layer now has "<<fnNeuronInLayer[1]<<" neurons"<<endl;
        }
        else{
          cout<<"Adding the last neuron didn't help significantly, not adding a neuron"<<endl;
        }
      }
      //*
      else{
        //cout<<"NO SIGNIFICANT IMPROVEMENT "<<previousFailureRate<<" :-> "<<worstFailureRate
         //   <<"(<"<<changeInFailureRateTolerance<<")after "<<fnConvergence<<" steps, quitting training!"<<endl;
       // break;
      }/**/
    }//end if no change in test result
    previousFailureRate=worstFailureRate;
  }while(!converged && fnConvergence<argMaxCount);
  if(converged)cout<<"Neural net converged after "<<fnConvergence<<" iterations, TPR, FPR = "<<fTPR<<" "<<fFPR<<" error vector norm "<<fErrorVectorNorm<<endl;
  else cout<<"Neural net did NOT converge after "<<fnConvergence<<" iterations, TPR, FPR = "<<fTPR<<" "<<fFPR<<" error vector norm "<<fErrorVectorNorm<<endl;

  return converged;
}















