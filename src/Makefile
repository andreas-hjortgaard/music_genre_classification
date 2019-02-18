SHARK_ROOT = /Users/andreashd/lib/Shark
BOOST_ROOT = /opt/local/include
BOOST_LIB  = /opt/local/lib
MARSYAS    = /Users/andreashd/lib/marsyas/build/
BOOSTF 		 = -I${BOOST_ROOT} -L${BOOST_LIB}
SHARKF		 = -I${SHARK_ROOT}/include -L${SHARK_ROOT}/lib/ 
MARSYASF	 = -I${MARSYAS}/include/marsyas -L${MARSYAS}/lib/ 
FRAMEWORKS = -framework CoreAudio -framework CoreMidi -framework CoreFoundation
LINKS			 = -lboost_serialization -lboost_system -lboost_filesystem -lboost_program_options
LINKS     += -lshark 
CPPFLAGS   = -DNDEBUG -DBOOST_UBLAS_NDEBUG -O3 #-g
FLAGS			 = -m64 # -Wall
ALLFLAGS	 = ${BOOSTF} ${SHARKF} ${MARSYASF} ${FRAMEWORKS} ${LINKS} ${CPPFLAGS} ${FLAGS}
OBJS			 = retrieveData.o featureLearning.o
CC         =  g++


all : learnfeatures genreclass

learnfeatures : retrieveData.o featureLearning.o 
	${CC} ${ALLFLAGS} ${OBJS} -lmarsyas learnfeatures.cpp -o learnfeatures

genreclass :
	${CC} ${BOOSTF} ${SHARKF} ${LINKS} ${CPPFLAGS} ${FLAGS} genreclass.cpp -o genreclass
 
retrieveData.o : retrieveData.h retrieveData.cpp
	${CC} ${BOOSTF} ${SHARKF} ${MARSYASF} ${CPPFLAGS} ${FLAGS} -c retrieveData.cpp

featureLearning.o : featureLearning.h featureLearning.cpp
	${CC} ${BOOSTF} ${SHARKF} ${MARSYASF} ${CPPFLAGS} ${FLAGS} -c featureLearning.cpp

clean:
	rm *.o learnfeatures genreclass