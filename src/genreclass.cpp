#include <iostream>
#include <vector>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h>

#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/Models/NearestNeighborClassifier.h>


using namespace std;
using namespace shark;

void usage(char *call) {
  fprintf(stdout, "Usage: %s path/to/training_set path/to/test_set\n", call);
}


int main(int argc, char **argv) {
  
  if (argc < 3) {
    usage(argv[0]);
    exit(1);
  }
  
  // training and test set paths
  string train_path(argv[1]);
  string test_path(argv[2]);
  
  // dataset
  LabeledData<RealVector, unsigned int> dataTrain;
  LabeledData<RealVector, unsigned int> dataTest;
  
  // import
  cout << "Importing data..." << endl;
  import_csv(dataTrain, train_path, FIRST_COLUMN, ",");
  import_csv(dataTest, test_path, FIRST_COLUMN, ",");
  
  // GENRE CLASSIFICATION WITH LDA AND KNN
  // TODO: Make majority vote classifier
  cout << "GENRE CLASSIFICATION WITH LDA AND KNN" << endl;
  
  Data<unsigned int> prediction;
  ZeroOneLoss<unsigned int> loss;
  
  // parameters
  const double lambda = 0.0;  // regularizer constant for LDA
  const unsigned int K = 1;   // K parameter for KNN
  
  cout << "lambda = " << lambda << endl;
  cout << "K = " << K << endl;
  
  // LDA classification
    
  LDA ldaTrainer(lambda);
  LinearClassifier lda(inputDimension(dataTrain), numberOfClasses(dataTrain));
  
  // train LDA
  cout << "Training LDA classifier..." << endl;
  ldaTrainer.train(lda, dataTrain);  
  
  // evaluate training error and test error
  prediction = lda.eval(dataTrain.inputs());
  cout << "LDA training error: " << loss.eval(dataTrain.labels(), prediction) << endl;
  prediction = lda.eval(dataTest.inputs());
  cout << "LDA test error: " << loss.eval(dataTest.labels(), prediction) << endl;
  
  
  // KNN classification 
  
  // prepare KNN
  LCTree<RealVector> *tree = new LCTree<RealVector>(dataTrain.inputs());
  
  // train KNN
  cout << "Creating KNN classifier..." << endl;
  NearestNeighborClassifier<RealVector> KNN = NearestNeighborClassifier<RealVector>(dataTrain, tree, K);
  
  // evaluate training error and test error
  prediction = KNN.eval(dataTrain.inputs());
  cout << "KNN training error: " << loss.eval(dataTrain.labels(), prediction) << endl;
  prediction = KNN.eval(dataTest.inputs());
  cout << "KNN test error: " << loss.eval(dataTest.labels(), prediction) << endl;
  

  return 0;
}