###################################################################
## A flexible method for genomics-based quantitative genetics in wild study systems – A case study on a house sparrow meta-population (2025)

## Paper authors: Anonymised for peer review
## Code author: Anonymised for peer review

## R-script that illustrates how to perform genomic prediction with the Bayesian Principal Component Ridge Regression (BPCRR) method, for the traits mass, wing length or tarsus length from the Helgeland house sparrow system. In this demonstration script we show how to fit the full model that contains a genetic effect, as well as additional fixed and random effects (formula 3 in the manuscript). In the case of BPCRR, the genetic term is approximated by a specified number of PCs that are derived from the SVD of the SNP matrix.

###################################################################

#################
### Loading necessary packages
#################
library(tictoc)
library(RSpectra)
library(Rfast)
library(missMethods)


# INLA is not a package on CRAN. It can be installed via the command 
# install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE) 
library(INLA) 


#################
### Setting some variables and loading data
#################

# Should svd be run? This parameter is 1 if you run the script for the first time, but may be set to 0 afterwards, so you avoid doing the svd calculations repeatedly.
do.svd=1

# Set a seed (you may try various ones)
seed <- 2024

# Loading the SNPs (3032 individuals with 180k SNPs each)
# d.SNP <- readRDS("SNP_180k_dummyRingRn.rds")

# Loading a small subset if you only want to see how the methods are used in practice
d.SNP <- readRDS("SNP_10k_dummyRingRn.rds")



# Loading and inspecting the morphology-file (d.morph) containing an individual ID (here stored as "ringnr"), fixed and random effects and the phenotypes body mass, wing length and tarsus length (denoted as mass, wing, tarsus)
load("d.morph_dummyRingRn.Rdata")
d.morph[1:5,]

# Decide here which of the three traits you want to perform the analysis on.
d.morph$trait <- d.morph$mass
# d.morph$trait <- d.morph$wing
# d.morph$trait <- d.morph$tarsus


# If we want to perform BPCRR with a fixed prior variance on the PC-effects (formula (4) in the paper), a prior guess for VA for the trait in interest are needed. Here we use informed guesses from earlier analyses. 

varA <- 1.3 # Body mass
# varA <- 1.6 # Wing length
# varA <- 0.3 # Tarsus length

# Before we continue with any further analysis, we will generate subsets of individuals that are both genotyped and have at least one measurement of the trait of interest (note that approximately 1/3 of the genotyped individuals do not have a measurement of the traits). Like this, we avoid doing the SVD of the SNP matrix for individuals that are not in the analysis. This step ensures that the PCs will not contain extra noise without explaining any of the variation in the traits.

####################################
# Now comes some data wrangling:
####################################
# First, a subset of individuals in d.morph having at least one measurement of e.g. body mass
d.morph_trait <- subset(d.morph, !is.na(d.morph$trait))

# Checking the number of individuals having at least one measurement of body trait, and making a vector of the unique ringnumbers
length(unique(d.morph_trait$ringnr))
ringnr <- unique(d.morph_trait$ringnr)
d.ringnr <- as.data.frame(ringnr)

# And selecting those individuals from the SNP matrix
d.SNP_trait <- merge(d.ringnr,d.SNP, by = "ringnr")

# This results in a matrix with ringnumbers (first column) and SNPs (all the remaining columns). We can see this by printing the upper left 10x10 block of the data:
d.SNP_trait[1:10,1:10]

# In order for INLA to fit the model, a sequential numerical ID in the same order as the genotype are needed. Making a vector of IDs for computations (IDC), and noting that there are fewer unique IDs in d.SNP_trait than in d.morph_trait
IDC <- c(1:length(d.SNP_trait$ringnr))

# Adding the IDCs as the first column in the SNP matrix and look at it again
d.SNP_trait <- cbind(IDC,d.SNP_trait)
d.SNP_trait[1:10,1:10]

# We now have a data frame with IDs and ringnumbers
ID_trait <- d.SNP_trait[,1:2]

# Merging with the morphology data frame for the trait of interest, and by keeping all rows in d.morph, all individuals will have a sequential numeric ID, and repeated measurements will all be tied to the same ID.
d.morph_trait <- merge(d.morph_trait,ID_trait, by = "ringnr", all.x = T)

# Keeping only the individuals where we have both a trait measurement and SNPs
d.morph_trait <- subset(d.morph_trait, !is.na(d.morph_trait$IDC))

# Adding an extra column with a copy of the IDC variable. This is a preparation for modelling multiple random effects, and is needed because INLA will not allow the same ID-name to be used twice in its formula. 
d.morph_trait$IDC1 <- d.morph_trait$IDC

# Some work still remains before the SNPs are ready fore use
# First remove the IDs and the ringnr from the SNP matrix
d.SNP_trait <- d.SNP_trait[,-c(1:2)]

# We extract the dimensions of the SNP matrix, which are the number of individuals (rows) and number of SNPs (rows)
dimensions <- dim(d.SNP_trait)
Nanimals <- as.numeric(dimensions[1])
nSNPs <- as.numeric(dimensions[2])

# checking for NAs
# percentage NAs in the matrix for this subsection
(sum(is.na(d.SNP_trait))/(dimensions[1]*dimensions[2]))*100

# Distribution of NAs
hist(colSums(is.na(d.SNP_trait)))
hist(rowSums(is.na(d.SNP_trait)))

# Maximum percentage of NAs in a SNP
(max(colSums(is.na(d.SNP_trait)))/dimensions[1])*100
# Maximum percentage of NAs in an individual.
(max(rowSums(is.na(d.SNP_trait)))/dimensions[2])*100

# Only a few values are missing, simple mode imputation using the function impute_mode() from the missMethods package will be performed before the SNPs now are ready for use.
d.SNP_trait <- impute_mode(d.SNP_trait, type = "columnwise")




#################
### Doing the SVD for the SNP matrix
#################

# Before doing the SVD, center the SNP matrix column-wise, so that mean(col)=0, but do not scale the variances to a standardized value:
d.SNP_trait_scaled <- scale(d.SNP_trait,center=TRUE,scale=FALSE)
d.SNP_trait_scaled <- as.matrix(d.SNP_trait_scaled)

# If you have already stored the SVD from an earlier run for the same trait, you can set do.svd=0 and thus do not need to re-run the code in the next lines.
# Note that a more efficient alternative to obtain the SVD is by using PLINK.


if (do.svd==1){
  k <- 12 # number of PCs to be estimated, default is # rows
  # Note that 1200 is the number of PCs estimated when doing the SVD - this number was chosen to "enough" PCs, but can be changed to a larger value, if needed.
  
  # Use the svds() function from the RSpectra package
  # Max iterations for our experiments was set to 2*10^6, instead of the default 1*10^6
  tmp.rspectra <- svds(d.SNP_trait_scaled, k, nu = k, nv = k, opts = list(maxitr = 2000000))
  
  # Inspection the cumulative variance op the PCs
  dd = prop.table(tmp.rspectra$d^2)
  plot(cumsum(dd))
}


#################
### Calculating the loadings matrix that will be used in the regression
#################
# This is in paper notation Z* = Z%*%V[,1:k] for the k selected in the SVD calculation:
XX  = d.SNP_trait_scaled %*% tmp.rspectra$v

# Scaling all PCs with the standard deviation of the first PC
XX <- XX*((1/sqrt(var(XX[,1]))))

# And centering (but not scaling the variances!) at the end of the preparation of the PCs
XX_c_trait <- scale(XX, center = TRUE, scale = FALSE)

# If you want to reduce the dimensionality of the PCs after initial analysis, this can be done after the scaling and centering to avoid running these operations multiple times. Note that speed decreases for larger n_PCs, while prediction accuracy improves. Using 1 000 PCs will be quite time consuming. To do a test run of the script, you can set a lower number of PCs, for example n_PCs <- 50.
n_PCs <- 10
XX_red <- XX[,1:n_PCs]
XX_c_trait <- XX_red


#################
### Derive the (fixed) prior that we give to the PC-effects-variance (formula (4) in the paper)
#################
# Summing the total variance of the PCs to use as fixed prior 
PCvar <- colVars(XX_red)
tot_PCvar <- sum(PCvar)

# As VA is set to a estimate of the traits VA, the fixed prior
u.prior.var <- varA/tot_PCvar




###################################################################
###########.         Genomic prediction.             ##############
###################################################################

# For illustration and efficiency, we are using a training and test set (not a 5- or 10-fold cross-validation). We choose 80% training and 20% test data. Note that we split the data individual-wise (not observation-wise) to ensure that the same individual cannot occur in both the training and test set.

# First we extract a list of unique ringnumbers
U_ringnr <- unique(d.morph_trait$ringnr)

set.seed(seed)
test.sample <- U_ringnr[sample(1:Nanimals,round(Nanimals*0.2),replace=FALSE)]

# We have to exclude the test data from the fitting procedure
# To this end, we are adding a new column in the data frame and replace all phenotypes for the individuals in the test set with NA. Note that INLA does then not use those rows to fit the model, but gives predictions for those entries, which is exactly what we want.

# To this end, we add a new column with the trait in interest for prediction and then set the phenotype to NA for the ringnumber in the test set
d.morph_trait$trait_test <- d.morph_trait$trait
d.morph_trait$trait_test[d.morph_trait$ringnr %in% test.sample] <- NA

# Checking the morphology file
d.morph_trait[1:20,]



############################################
### INLA setup for BPCRR with fixed prior
############################################

# To implement ridge-type shrinkage in INLA, we are making use of the so-called "z-model", which is INLA's way to model random effects with ridge shrinkage (See here https://inla.r-inla-download.org/r-inla.org/doc/latent/z.pdf). We also refer to the book "Bayesian Regression Moldeling with INLA" by Wang, Yue and Faraway (2018), chapter 5.4.1.

# The formula for the regression is given as
formula.trait.fixed = trait_test~   sex + FGRM + month + age + outer + other +  
  f(hatchyear,model="iid",hyper=list(
    prec=list(initial=log(10), prior="pc.prec",param=c(1,0.05)) # Prior for the hatch year variance. An initial value is given to inla to ensure the algorithm iterates around reasonable values. Parametrization is as log(1/variance). Here, the initial value for the variance is thus 1/10.
  ))+
  f(IDC,model="iid",hyper=list(
    prec=list(initial=log(1), prior="pc.prec",param=c(1,0.05)) # Prior for the ID-effect variance. Initial value for the variance is 1.
  ))+
  f(island_current,model="iid",hyper=list(
    prec=list(initial=log(10), prior="pc.prec",param=c(1,0.05)) # Prior for the island variance. Initial value for the variance is 1/10.
  ))+
  # The last part corresponds to the genomic part with the PCs as random variables and ridge shrinkage
  f(IDC1, model = "z", Z = XX_c_trait,
    hyper=list(
      prec=list(initial=log(1/u.prior.var),
                fixed=TRUE
                # fixed=TRUE fixes the variance at u.prior.var 
                # fixed=FALSE would give the default priors, but other priors can be specified as well.
      ) 
    )
  )


# Now we are calling inla() - this call takes a bit of time (use tic() and toc() from the tictoc package to measure):
tic()
model.trait.fixed = inla(formula=formula.trait.fixed, family="gaussian",
                         data=d.morph_trait,
                         control.family=list(hyper = list(theta = list(initial=log(1),
                                                                       prior="pc.prec",
                                                                       param=c(1,0.05)))), 
                         control.compute=list(dic=F, config = TRUE,
                                              return.marginals=FALSE), # To be able to resample from the inla object, we need config=TRUE. However, config=FALSE makes the computation faster, so only set it to TRUE if you plan to resample (i.e., if you want to estimate VA)
                         num.threads=8 # Set the number of cores for parallel computation. Use 1-2 less than what you have on your machine. 
)
toc()

# In case you are unsure whether the initial values given to the variance parameters in the formula above were good, you can re-run inla once, using the estimated values from the first run as new initial values.
#model.trait.fixed <- inla.rerun(model.trait.fixed)  


#####################
# Genomic prediction and assessment of accuracy using the output from INLA
#####################

# We use the posterior mean as predicted breeding values for all individuals (both training and test set)
breedingv <- model.trait.fixed$summary.random$IDC1$mean[1:Nanimals]

# As we have one breedingvalue for each individual, but possibly multiple measurements of the same individual, some more wrangling are necessary.
# A vector of IDs
IDC <- c(1:Nanimals)

# We create a new data frame where the first column is the individual ID (IDC), and the second column is the corresponding predicted breeding value:
bvAndIds <- as.data.frame(IDC)
bvAndIds$breedingv <- breedingv

# Merge phenotypes and predicted breeding values for all individuals into one data frame
d.morph_and_breedingv <- merge(x=d.morph_trait,y=bvAndIds,by="IDC",all.x=TRUE)

# Subset containing only the individuals where the trait was predicted (corresponds to the responses that were NA in the trait_test variable)
d.morph_and_breedingv <- d.morph_and_breedingv[d.morph_and_breedingv$ringnr %in% test.sample,]

# Generating the mean phenotypes and predicted breeding values (breeding values are all the same for each measurement for the individual, just a hack in order to match them up to the corresponding mean phenotype) for each individual 
mean_traitByIDs <- aggregate(trait~IDC, data=d.morph_and_breedingv, mean, na.action = na.pass)
breedingvByIDs <- aggregate(breedingv~IDC, data=d.morph_and_breedingv, mean)

plot(breedingvByIDs$breedingv ~ mean_traitByIDs$trait,
     xlab="phenotype",
     ylab="predicted breeding value",
     main="INLA")

# And finally the correlation between the predicted breeding value and the actual phenotypes that were spared out in the respective test sample:
cor_phenobreed_trait_fixed <- cor(breedingvByIDs$breedingv , mean_traitByIDs$trait,use="complete.obs")
print(cor_phenobreed_trait_fixed)
