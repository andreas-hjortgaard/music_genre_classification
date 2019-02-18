#include <iostream>
#include <vector>
#include <MarSystemManager.h>
#include <SoundFileSource.h>
#include <shark/Data/Dataset.h>
#include <boost/tokenizer.hpp>

#include "retrieveData.h"

using namespace std;
using namespace Marsyas;
using namespace shark;

// helper functions
RealVector mrs_realvecToRealVector(mrs_realvec *rv) {

    int N = rv->getSize();
    RealVector data(N);

    // set each vector entry to the corresponding value from the mrs_realvec
    for (int i = 0; i < N; i++) {
        data(i) = rv->getValueFenced(i);
    }

    return data;
}

vector<string> split(string str) {

    vector<string> str_split;
    boost::char_separator<char> sep(", ");
    boost::tokenizer<boost::char_separator<char> > tokens(str, sep);
    boost::tokenizer<boost::char_separator<char> >::iterator tok_iter;

    for (tok_iter = tokens.begin(); tok_iter != tokens.end(); tok_iter++) {
        str_split.push_back(*tok_iter);
    }

    return str_split;
}

unsigned int convertLabel(mrs_real label) {
  return static_cast<unsigned int>(label);
}


mdata retrieveData(string collection, int frames) {

    // create manager for a series type flow
    MarSystemManager mng;
    mrs_realvec raw;
    mrs_realvec spectrum;
    mrs_realvec mfcc;
    mrs_natural label;
    RealVector raw_rv;
    RealVector spectrum_rv;
    RealVector mfcc_rv;
    MarSystem *read_data = mng.create("Series", "read_data");

    // audio source
    read_data->addMarSystem(mng.create("SoundFileSource", "src"));
    read_data->updControl("SoundFileSource/src/mrs_string/filename", collection);

    // sliding window
    read_data->addMarSystem(mng.create("ShiftInput", "frames"));
    read_data->updControl("ShiftInput/frames/mrs_natural/winSize", 1024);

    // apply Hamming window
    read_data->addMarSystem(mng.create("Windowing", "hamming"));
    read_data->updControl("Windowing/hamming/mrs_string/type", "Hamming");

    // compute FFT spectrum
    read_data->addMarSystem(mng.create("Spectrum","spectrum"));

    // compute magnitude spectrum
    read_data->addMarSystem(mng.create("PowerSpectrum","power"));
    read_data->updControl("PowerSpectrum/power/mrs_string/spectrumType", "magnitude");

    // compute MFCCs
    read_data->addMarSystem(mng.create("MFCC","mfcc"));
    read_data->updControl("MFCC/mfcc/mrs_natural/coefficients", 5);

    // dummy classification for extracting MFCCs
    read_data->addMarSystem(mng.create("KNNClassifier","knn"));

    int num_files = read_data->getControl("SoundFileSource/src/mrs_natural/numFiles")->to<mrs_natural>();
    UnlabeledData<RealVector> unlabeled_raw(num_files*frames);
    UnlabeledData<RealVector> unlabeled_spectrum(num_files*frames);
    UnlabeledData<RealVector> unlabeled_mfcc(num_files*frames);
    Data<unsigned int> labels(num_files*frames);
    int index = 0;

    // read a number of frames into a RealVector and add each song to the UnlabeledData structure
    for (int i = 0; i < num_files; i++) {
        for (int j = 0; j < frames; j++) {

            read_data->tick();

            // get spectrum data, MFCCs and labels
            raw       = read_data->getControl("Windowing/hamming/mrs_realvec/processedData")->to<mrs_realvec>();
            spectrum  = read_data->getControl("PowerSpectrum/power/mrs_realvec/processedData")->to<mrs_realvec>();
            mfcc      = read_data->getControl("MFCC/mfcc/mrs_realvec/processedData")->to<mrs_realvec>();
            label     = read_data->getControl("SoundFileSource/src/mrs_real/currentLabel")->to<mrs_real>();

            // convert to RealVector
            raw_rv = mrs_realvecToRealVector(&raw);
            spectrum_rv = mrs_realvecToRealVector(&spectrum);
            mfcc_rv = mrs_realvecToRealVector(&mfcc);

            // insert into data structures
            unlabeled_raw.setElement(index, raw_rv);
            unlabeled_spectrum.setElement(index, spectrum_rv);
            unlabeled_mfcc.setElement(index, mfcc_rv);
            labels.setElement(index, convertLabel(label));

            index++;
        }
        read_data->updControl("SoundFileSource/src/mrs_natural/advance", 1);
    }

    // create labeled data vector
    UnlabeledData<RealVector> raw_data(unlabeled_raw);
    UnlabeledData<RealVector> spectrum_data(unlabeled_spectrum);
    UnlabeledData<RealVector> mfcc_data(unlabeled_mfcc);

    // create data struct
    mdata data;
    data.raw      = raw_data;
    data.spectrum = spectrum_data;
    data.mfcc     = mfcc_data;
    data.labels   = labels;


    delete read_data;

    return data;

}

// return the label names
vector<string> retrieveLabelNames(string collection) {

    MarSystemManager mng;
    vector<string> label_names;
    MarSystem *read_labels = mng.create("Series", "label_names");

    // collection source
    read_labels->addMarSystem(mng.create("SoundFileSource", "src"));
    read_labels->updControl("SoundFileSource/src/mrs_string/filename", collection);

    // retrieve labels
    label_names = split(read_labels->getControl("SoundFileSource/src/mrs_string/labelNames")->to<string>());

    return label_names;
}
