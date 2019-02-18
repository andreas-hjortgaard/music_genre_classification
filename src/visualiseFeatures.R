# script for visualising features using t-SNE
require(tsne)

# read data
#raw <- read.table("features/201110282236/raw_train.data", sep=",")
#raw_labels <- as.factor(raw[,1])
#raw <- raw[,-1] 

all <- TRUE

spectrum <- read.table("features/20111104/spectrum_train.data", sep=",")

# randomly select 500 frames
indices <- 1:length(spectrum[,1])
selected <- sample(indices, 500)

spectrum_labels <- as.factor(spectrum[selected,1])
spectrum <- spectrum[selected,-1] 

mfcc <- read.table("features/20111104/mfcc_train.data", sep=",")
mfcc_labels <- as.factor(mfcc[selected,1])
mfcc <- mfcc[selected,-1]

rbm <- read.table("features/20111104/rbm_train.data", sep=",")
rbm_labels <- as.factor(rbm[selected,1])
rbm <- rbm[selected,-1]


# visualise data examples (unused...but interesting)
pdf("features/20111104/spectrum_rep.pdf")
plot(t(spectrum[1,]), type="l", ylab="magnitude")
dev.off()

pdf("features/20111104/mfcc_rep.pdf")
plot(t(mfcc[1,]), type="l" , ylab="amplitude")
dev.off()
  
pdf("features/20111104/rbm_rep.pdf")
plot(t(rbm[1,]), type="l", ylab="magnitude")
dev.off()


# perform dimensionality reduction
if (all) {
  redux_spectrum <- tsne(spectrum, k=2, max_iter=1000, epoch=100)
  redux_mfcc <- tsne(mfcc, k=2, max_iter=1000, epoch=100)
}
redux_rbm <- tsne(rbm, k=2, max_iter=1000, epoch=100)


# visualise with colour codings

genres <- c("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")
gcolors <- c("black", "blue", "blue", "black", "purple", "cyan", "orange", "orange", "red", "green") 
gchars <- c(4, 6, 15, 3, 17, 17, 19, 17, 16, 1)

if (all) { 
# spectrum
g0 <- which(spectrum_labels == 0)   # blues
g1 <- which(spectrum_labels == 1)   # classical
g2 <- which(spectrum_labels == 2)   # country
g3 <- which(spectrum_labels == 3)   # disco
g4 <- which(spectrum_labels == 4)   # hiphop
g5 <- which(spectrum_labels == 5)   # jazz
g6 <- which(spectrum_labels == 6)   # metal
g7 <- which(spectrum_labels == 7)   # pop
g8 <- which(spectrum_labels == 8)   # reggae
g9 <- which(spectrum_labels == 9)   # rock

pdf("features/20111104/spectrum_redux.pdf")

plot(redux_spectrum$ydata[g0,], col="black", pch=4, xlab="", ylab="")
points(redux_spectrum$ydata[g1,], col="blue", pch=6)
points(redux_spectrum$ydata[g2,], col="blue", pch=15)
points(redux_spectrum$ydata[g3,], col="black", pch=3)
points(redux_spectrum$ydata[g4,], col="purple", pch=17)
points(redux_spectrum$ydata[g5,], col="cyan", pch=17)
points(redux_spectrum$ydata[g6,], col="orange", pch=19)
points(redux_spectrum$ydata[g7,], col="orange", pch=17)
points(redux_spectrum$ydata[g8,], col="red", pch=16)
points(redux_spectrum$ydata[g9,], col="green", pch=1)

dev.off()

# MFCC 
g0 <- which(mfcc_labels == 0)
g1 <- which(mfcc_labels == 1)
g2 <- which(mfcc_labels == 2)
g3 <- which(mfcc_labels == 3)
g4 <- which(mfcc_labels == 4)
g5 <- which(mfcc_labels == 5)
g6 <- which(mfcc_labels == 6)
g7 <- which(mfcc_labels == 7)
g8 <- which(mfcc_labels == 8)
g9 <- which(mfcc_labels == 9)

pdf("features/20111104/mfcc_redux.pdf")

plot(redux_mfcc$ydata[g0,], col="black", pch=4, xlab="", ylab="")
points(redux_mfcc$ydata[g1,], col="blue", pch=6)
points(redux_mfcc$ydata[g2,], col="blue", pch=15)
points(redux_mfcc$ydata[g3,], col="black", pch=3)
points(redux_mfcc$ydata[g4,], col="purple", pch=17)
points(redux_mfcc$ydata[g5,], col="cyan", pch=17)
points(redux_mfcc$ydata[g6,], col="orange", pch=19)
points(redux_mfcc$ydata[g7,], col="orange", pch=17)
points(redux_mfcc$ydata[g8,], col="red", pch=16)
points(redux_mfcc$ydata[g9,], col="green", pch=1)

dev.off()

}

# RBM
g0 <- which(rbm_labels == 0)
g1 <- which(rbm_labels == 1)
g2 <- which(rbm_labels == 2)
g3 <- which(rbm_labels == 3)
g4 <- which(rbm_labels == 4)
g5 <- which(rbm_labels == 5)
g6 <- which(rbm_labels == 6)
g7 <- which(rbm_labels == 7)
g8 <- which(rbm_labels == 8)
g9 <- which(rbm_labels == 9)

pdf("features/20111104/rbm_redux.pdf")

plot(redux_rbm$ydata[g0,], col="black", pch=4, xlab="", ylab="")
points(redux_rbm$ydata[g1,], col="blue", pch=6)
points(redux_rbm$ydata[g2,], col="blue", pch=15)
points(redux_rbm$ydata[g3,], col="black", pch=3)
points(redux_rbm$ydata[g4,], col="purple", pch=17)
points(redux_rbm$ydata[g5,], col="cyan", pch=17)
points(redux_rbm$ydata[g6,], col="orange", pch=19)
points(redux_rbm$ydata[g7,], col="orange", pch=17)
points(redux_rbm$ydata[g8,], col="red", pch=16)
points(redux_rbm$ydata[g9,], col="green", pch=1)

dev.off()


# legend

pdf("features/20111104/legend.pdf", width=3, height=5)
# empty plot
plot(1, type="n", axes=F, xlab="", ylab="")
legend("left", legend=genres, col=gcolors, pch=gchars, title="genres")
dev.off()



