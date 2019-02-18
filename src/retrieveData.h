/* 
 * Functions for reading data vectors (raw audio features)
 */ 
#include <string>
#include <vector>
#include <MarSystemManager.h>
#include <shark/Data/Dataset.h>

// structure for keeping the musical data retrieved from collection
struct mdata {
  shark::UnlabeledData<shark::RealVector> raw;
  shark::UnlabeledData<shark::RealVector> spectrum;
  shark::UnlabeledData<shark::RealVector> mfcc;
  shark::Data<unsigned int> labels;
};

// returns vectors of size 513 for each N frames for each song along with the class label
mdata retrieveData(std::string collection, int frames);


// returns the label names for each class (genres) 
std::vector<std::string> retrieveLabelNames(std::string collection);