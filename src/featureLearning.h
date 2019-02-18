#include <vector>
#include <shark/Data/Dataset.h>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/RBM.h>
#include <shark/Unsupervised/RBM/Energy.h>


#include <shark/Unsupervised/RBM/Neuronlayers/TruncatedExponentialLayer.h>
#include <shark/Unsupervised/RBM/Sampling/MarkovChain.h>
#include <shark/Unsupervised/RBM/Sampling/TemperedMarkovChain.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>

#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Unsupervised/RBM/GradientApproximations/ContrastiveDivergence.h>
#include <shark/Unsupervised/RBM/GradientApproximations/MultiChainApproximator.h>
#include <shark/Unsupervised/RBM/GradientApproximations/SimpleApproximator.h>


#include <shark/Rng/GlobalRng.h>

using namespace shark;

// define Truncated Exponential RBM
typedef Energy<TruncatedExponentialLayer, BinaryLayer> ExpEnergy;
typedef RBM<ExpEnergy, Rng::rng_type> ExpRBM;
typedef GibbsOperator<ExpRBM> ExpGibbsOperator;
typedef MarkovChain<ExpGibbsOperator> ExpGibbsChain;
typedef TemperedMarkovChain<ExpGibbsOperator> ExpPTChain;

typedef MultiChainApproximator<ExpGibbsChain> ExpPCD;
typedef ContrastiveDivergence<ExpGibbsOperator> ExpCD;
typedef GradientApproximator<ExpPTChain> ExpParallelTempering;


// use these typedefs to switch between ExpRBM and BinaryRBM
typedef ExpRBM MusicalRBM;
typedef ExpCD MusicalCD;

// learn features using RBM
MusicalRBM learnFeatures( UnlabeledData<RealVector> &data,     // musical training set (usually magnitude spectrum)
                          size_t numberOfHidden,              // dimensionality of the learned features
                          unsigned int numIterations,         // number of iterations for training
                          unsigned int numTrials,             // number of trials over the training set
                          unsigned int k = 1,                 // number of Markov chain steps in CD training (default: 1)
                          int optMomentum = 0,                // optimizer: momentum (default: 0)
                          float optLearningRate = 0.1);       // optimizer: learning rate (default: 0.1)

// convert data to hidden layer representation
UnlabeledData<RealVector> convertToRBMRepresentation(const MusicalRBM &rbm, UnlabeledData<RealVector> data);
