#include <iostream>
#include <vector>
#include <MarSystemManager.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>

#include "retrieveData.h"
#include "featureLearning.h"

using namespace std;
using namespace shark;
using namespace Marsyas;

void usage(char *call) {
  fprintf(stdout, "Usage: %s /path/to/collection [path/to/rbm_train_features path/to/rbm_test_features]\n", call);
}

// data structure for holding all the datasets
struct UDatasets {
  
  UnlabeledData<RealVector> dataTrainRawU;
  UnlabeledData<RealVector> dataTestRawU;
  
  UnlabeledData<RealVector> dataTrainSpectrumU;
  UnlabeledData<RealVector> dataTestSpectrumU;
  
  UnlabeledData<RealVector> dataTrainMFCCU; 
  UnlabeledData<RealVector> dataTestMFCCU;
  
  UnlabeledData<RealVector> dataTrainRBMU; 
  UnlabeledData<RealVector> dataTestRBMU;
  
  UnlabeledData<unsigned int> labelsTrain; 
  UnlabeledData<unsigned int> labelsTest;
};


// retrieve raw, spectrum and MFCC features 
void retrieveDatasets(UDatasets &datasets, string collection, int frames, float trainingPercentage) {
  
  mdata data; // struct that keeps both spectrum features and MFCC features for each frame
  vector<string> labelNames;
  
  cout << "Reading files..." << endl;
  
  data = retrieveData(collection, frames);
  labelNames = retrieveLabelNames(collection);
  
  // print genres
  for (vector<string>::iterator i = labelNames.begin(); i != labelNames.end(); i++) {
    cout << *i << endl;
  }
  
  // create training and test data from the spectrum data
  UnlabeledData<RealVector> dataTrainRawU, dataTestRawU;
  UnlabeledData<RealVector> dataTrainSpectrumU, dataTestSpectrumU;
  UnlabeledData<RealVector> dataTrainMFCCU, dataTestMFCCU;
  UnlabeledData<unsigned int> labelsTrain, labelsTest;
  
  unsigned int trainingSize = data.spectrum.size()*trainingPercentage;
  cout << "Training set size: " << trainingSize << endl;
  
  // construct training and test sets for all features
  vector<size_t> indices;
  detail::random(trainingSize, data.spectrum.size(), indices);
  data.raw.indexedSubset(indices, dataTrainRawU, dataTestRawU);
  data.spectrum.indexedSubset(indices, dataTrainSpectrumU, dataTestSpectrumU);
  data.mfcc.indexedSubset(indices, dataTrainMFCCU, dataTestMFCCU);
  data.labels.indexedSubset(indices, labelsTrain, labelsTest);
  
  
  // NORMALIZE DATA
  cout << "Normalizing data..." << endl;
  NormalizeComponentsUnitInterval<> normalizingTrainer;
  size_t numberOfVisible; 
  
  // raw data
  numberOfVisible = dataTrainRawU.element(0).size();
  LinearModel<> normalizerRaw(numberOfVisible, numberOfVisible, true);
  normalizingTrainer.train(normalizerRaw, dataTrainRawU.inputs());  
  
  dataTrainRawU.transform(normalizerRaw);
  dataTestRawU.transform(normalizerRaw);
  
  // spectrum data
  numberOfVisible = dataTrainSpectrumU.element(0).size();
  LinearModel<> normalizerSpectrum(numberOfVisible, numberOfVisible, true);
  normalizingTrainer.train(normalizerSpectrum, dataTrainSpectrumU.inputs());  
  
  dataTrainSpectrumU.transform(normalizerSpectrum);
  dataTestSpectrumU.transform(normalizerSpectrum);
  
  // MFCC data
  numberOfVisible = dataTrainMFCCU.element(0).size();
  LinearModel<> normalizerMFCC(numberOfVisible, numberOfVisible, true);
  normalizingTrainer.train(normalizerMFCC, dataTrainMFCCU.inputs());  
  
  dataTrainMFCCU.transform(normalizerMFCC);
  dataTestMFCCU.transform(normalizerMFCC);
  
  
  // insert into structure
  datasets.dataTrainRawU = dataTrainRawU;
  datasets.dataTestRawU = dataTestRawU;
  
  datasets.dataTrainSpectrumU = dataTrainSpectrumU;
  datasets.dataTestSpectrumU = dataTestSpectrumU;
  
  datasets.dataTrainMFCCU = dataTrainMFCCU;
  datasets.dataTestMFCCU = dataTestMFCCU;  
  
  datasets.labelsTrain = labelsTrain; 
  datasets.labelsTest = labelsTest;
}



// learn RBM features from magnitude spectrum 
void learnRBMFeatures(UDatasets &datasets, size_t numberOfHidden, unsigned int numberOfIterations, unsigned int numberOfTrials) {
  
  cout << "Training RBM..." << endl; 
  
  // TODO: Use of MFCC or spectrum must be a parameter
  MusicalRBM rbm = learnFeatures(datasets.dataTrainSpectrumU, numberOfHidden, numberOfIterations, numberOfTrials);
  cout << "Done!" << endl;
  
  // create RBM representation of data
  UnlabeledData<RealVector> dataTrainRBMU = convertToRBMRepresentation(rbm, datasets.dataTrainSpectrumU);
  UnlabeledData<RealVector> dataTestRBMU = convertToRBMRepresentation(rbm, datasets.dataTestSpectrumU);
  cout << "RBM representation: " << dataTrainRBMU.element(0).size() << endl;
  
  // normalize
  NormalizeComponentsUnitInterval<> normalizingTrainer;
  size_t numberOfVisible; 
  
  numberOfVisible = dataTrainRBMU.element(0).size();
  LinearModel<> normalizerRBM(numberOfVisible, numberOfVisible, true);
  normalizingTrainer.train(normalizerRBM, dataTrainRBMU.inputs());  
  
  dataTrainRBMU.transform(normalizerRBM);
  dataTestRBMU.transform(normalizerRBM);
    
  // insert into structure
  datasets.dataTrainRBMU = dataTrainRBMU;
  datasets.dataTestRBMU = dataTestRBMU;
  
} 


// main function that retrieves datasets, learns features, store features for visualisation and 
// performs genre classification
int main(int argc, char **argv) {
  
  bool learn_features = true;
  string rbm_train, rbm_test;
  
  // collection must be given as argument or three (four with call) arguments must be given
  if (argc < 2 || argc == 3) {
    usage(argv[0]);
    exit(1);
  }
  
  // first argument: collection
  string collection(argv[1]);
  cout << "Collection: " << collection << endl;
  
  // second and third argument (optional): RBM features
  if (argc > 2) {
    rbm_train         = argv[2];
    rbm_test          = argv[3];
    learn_features    = false;
  }
  
  
  
  // --- RETRIEVE DATA ---
  int frames = 200;
  float trainingPercentage = 0.7;
  
  UDatasets datasets;
  retrieveDatasets(datasets, collection, frames, trainingPercentage);
  
  
  // --- LEARN RBM FEATURES ---
  if (learn_features) {
    size_t numberOfHidden           = 50;
    unsigned int numberOfIterations = 1000;
    unsigned int numberOfTrials     = 10;
  
    learnRBMFeatures(datasets, numberOfHidden, numberOfIterations, numberOfTrials);
  
  }
    
    
  // --- CREATE LABELED DATASETS AND STORE FOR VISUALISATION---
  
  cout << "Store features for visualisation and classification..." << endl;  
  
  // RBM training and test
  LabeledData<RealVector, unsigned int> dataTrainRBM;
  LabeledData<RealVector, unsigned int> dataTestRBM;
  if (learn_features) {
    dataTrainRBM = LabeledData<RealVector, unsigned int>(datasets.dataTrainRBMU, datasets.labelsTrain);
    dataTestRBM = LabeledData<RealVector, unsigned int>(datasets.dataTestRBMU, datasets.labelsTest);
    export_csv(dataTrainRBM, "features/rbm_train.data", FIRST_COLUMN, ",");
    export_csv(dataTestRBM, "features/rbm_test.data", FIRST_COLUMN, ",");
  } else {
    // if no features have been learned they must be imported 
    import_csv(dataTrainRBM, rbm_train, FIRST_COLUMN, ",");
    import_csv(dataTestRBM, rbm_test, FIRST_COLUMN, ",");
  }
  
  
  // raw training and test
  LabeledData<RealVector, unsigned int> dataTrainRaw(datasets.dataTrainRawU, datasets.labelsTrain);
  LabeledData<RealVector, unsigned int> dataTestRaw(datasets.dataTestRawU, datasets.labelsTest);
  export_csv(dataTrainRaw, "features/raw_train.data", FIRST_COLUMN, ",");
  export_csv(dataTestRaw, "features/raw_test.data", FIRST_COLUMN, ",");
  
  // spectrum training and test
  LabeledData<RealVector, unsigned int> dataTrainSpectrum(datasets.dataTrainSpectrumU, datasets.labelsTrain);
  LabeledData<RealVector, unsigned int> dataTestSpectrum(datasets.dataTestSpectrumU, datasets.labelsTest);
  export_csv(dataTrainSpectrum, "features/spectrum_train.data", FIRST_COLUMN, ",");
  export_csv(dataTestSpectrum, "features/spectrum_test.data", FIRST_COLUMN, ",");
  
  // MFCC training and test
  LabeledData<RealVector, unsigned int> dataTrainMFCC(datasets.dataTrainMFCCU, datasets.labelsTrain);
  LabeledData<RealVector, unsigned int> dataTestMFCC(datasets.dataTestMFCCU, datasets.labelsTest);
  export_csv(dataTrainMFCC, "features/mfcc_train.data", FIRST_COLUMN, ",");
  export_csv(dataTestMFCC, "features/mfcc_test.data", FIRST_COLUMN, ",");
  
  cout << "Done!" << endl;  
  
  return 0;
  
}