#include <vector>
#include <shark/Rng/GlobalRng.h>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Models/FFNet.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include "featureLearning.h"

using namespace std;
using namespace shark;


// initialize weights of rbm to be between -0.1 and 0.1
void initializeWeights(MusicalRBM &rbm) {

    RealVector weights(rbm.numberOfParameters());

    for (size_t i = 0; i != weights.size(); i++) {
        weights(i) = Rng::uni(-0.1,0.1);
    }

    rbm.setParameterVector(weights);
}

// learn features
MusicalRBM learnFeatures( UnlabeledData<RealVector> &data,     // musical training set (usually magnitude spectrum)
                          size_t numberOfHidden,              // dimensionality of the learned features
                          unsigned int numIterations,         // number of iterations for training
                          unsigned int numTrials,             // number of trials over the training set
                          unsigned int k,                     // number of Markov chain steps in CD training (default: 1)
                          int optMomentum,                    // optimizer: momentum (default: 0)
                          float optLearningRate)              // optimizer: learning rate (default: 0.1)
{


    // setup RBM
    size_t numberOfVisible = data.element(0).size();   //visible units of the inputs

    // create binary RBM
    MusicalRBM rbm(Rng::globalRng);
    rbm.setStructure(numberOfVisible, numberOfHidden);

    // setup optimizer
    SteepestDescent optimizer;
    optimizer.setMomentum(optMomentum);
    optimizer.setLearningRate(optLearningRate);

    // contrastive divergence
    MusicalCD cd(&rbm);
    cd.setK(k);
    cd.setData(data);

    cout << "Number of visible: " << numberOfVisible << endl;
    cout << "Number of hidden: " << numberOfHidden << endl;

    // train RBM with CD
    for (unsigned int trial = 0; trial < numTrials; trial++) {
        initializeWeights(rbm);
        optimizer.init(cd);

        cout << "Trial: " << trial+1 << " of " << numTrials << endl;

        for (unsigned int iteration = 0; iteration < numIterations; iteration++) {

            cout << "Iteration: " << iteration+1 << " of " << numIterations << endl;
            optimizer.step(cd);

        }
    }

    return rbm;
}


// converts a dataset to the corresponding hidden layer RBM representation
UnlabeledData<RealVector> convertToRBMRepresentation(const MusicalRBM &rbm, UnlabeledData<RealVector> data) {

    FFNet<Fermi,Fermi> net;
    net.setStructure(rbm.numberOfVN(), 0, rbm.numberOfHN());
    net.layerMatrices()[0] = rbm.structure().weightMatrix(0,0);
    subrange(net.bias(), rbm.numberOfVN(), rbm.numberOfVN()+rbm.numberOfHN()) = rbm.hiddenNeurons().bias();

    UnlabeledData<RealVector> new_representation = data;
    new_representation.transform(net);

    return new_representation;

}
