#include "NNData.hh"


NNData::NNData():fVersion(0), fInput(0),fInputNorm(0), fOutput(0), fOutputNorm(0),fInputNormByEntry(0)
{


}

NNData::~NNData()
{


}

std::ostream& operator<<(std::ostream& argOStream, NNData* argNNData)
{
  if(argNNData->fVersion==0){
    argOStream<<argNNData->fVersion<<" "<<argNNData->GetnData()<<" "<<argNNData->fInput[0].size()<<" ";
    argOStream<<argNNData->fOutput.size()<<" ";
    if(argNNData->fOutput.size()>0)argOStream<<argNNData->fOutput[0].size()<<" ";
    argOStream<<" "<<argNNData->fInputNorm.size()<<" "<<argNNData->fOutputNorm.size()<<endl;

    for(size_t i=0;i<argNNData->fInputNorm.size();i++){
      argOStream<<argNNData->fInputNorm[i]<<" ";
    }
    for(size_t i=0;i<argNNData->fOutputNorm.size();i++){
      argOStream<<argNNData->fOutputNorm[i]<<" ";
    }

    argOStream<<endl;

    for(size_t i=0;i<argNNData->GetnData();i++){
        argOStream<<argNNData->fInputNormByEntry[i]<<" ";
      for(size_t j=0;j<argNNData->fInput[0].size();j++){
        argOStream<<argNNData->fInput[i][j]<<" ";
      }
      if(argNNData->fOutput.size()>0){
          for(size_t j=0;j<argNNData->fOutput[0].size();j++){
          argOStream<<argNNData->fOutput[i][j]<<" ";
        }
      }
      argOStream<<endl;
    }

  }
  else{cout<<"ERROR Version for file output not recognized"<<endl;}
  return argOStream;
}

std::istream& operator>>(std::istream& argIStream, NNData* argNNData)
{
  argIStream>>argNNData->fVersion;
  if(argNNData->fVersion==0){
    size_t nData=0;
    size_t nInput=0;
    size_t nDataO;
    size_t nOutput=0;
    size_t nOutputNorm=0;
    size_t nInputNorm=0;

    argIStream>>nData>>nInput;
    argIStream>>nDataO;
    if(nDataO>0)argIStream>>nOutput;
    argIStream>>nInputNorm>>nOutputNorm;

    argNNData->fInputNorm.resize(nInputNorm);
    argNNData->fOutputNorm.resize(nOutputNorm);

    for(size_t i=0;i<argNNData->fInputNorm.size();i++){
      argIStream>>argNNData->fInputNorm[i];
    }
    for(size_t i=0;i<argNNData->fOutputNorm.size();i++){
      argIStream>>argNNData->fOutputNorm[i];
    }

    argNNData->fInput.resize(nData);
    argNNData->fInputNormByEntry.resize(nData);
    argNNData->fOutput.resize(nDataO);

    for(size_t i=0;i<argNNData->GetnData();i++){
      argNNData->fInput[i].resize(nInput);
      argIStream>>argNNData->fInputNormByEntry[i];
      for(size_t j=0;j<argNNData->fInput[0].size();j++){
        argIStream>>argNNData->fInput[i][j];
      }
      if(nDataO>0){
        argNNData->fOutput[i].resize(nOutput);
        for(size_t j=0;j<argNNData->fOutput[0].size();j++){
          argIStream>>argNNData->fOutput[i][j];
        }
      }
    }
  }
  else{cout<<"ERROR Version for file input not recognized"<<endl;}
  return argIStream;
}





