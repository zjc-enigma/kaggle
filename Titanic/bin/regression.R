rm(list=ls())
require("plyr")
require("Amelia")
require("vcd")
require("corrgram")
require("stringr")
require("caret")
require("ada")


options(digits=2)

train.file <- "../data/train.csv"
test.file <- "../data/test.csv"

train.data <- read.csv(file=train.file,
                       sep=',',
                       stringsAsFactors=F,
                       na.strings=c("NA", ""))

missmap(train.data,
        main="Titanic training data - missmap",
        col=c("red", "black"),
        legend=FALSE)

# explore data

barplot(table(train.data$Survived),
        names.arg=c("perished", "survived"),
        main="survived", col="black")

barplot(table(train.data$Pclass),
        names.arg=c("1st", "2nd", "3rd"),
        main="Pclass", col="firebrick"
        )

barplot(table(train.data$Sex),
         main="Sex", col="darkviolet"
        )

hist(train.data$Age,
     main="Age distribution",
     xlab=NULL,
     col="brown"
     )

barplot(table(train.data$SibSp),
        main="SibSp (siblings + spouse aboard)",
        col="darkblue")

barplot(table(train.data$Parch),
        main="Parch (parents + kids aboard)",
        col="gray50"
        )

hist(train.data$Fare,
     main="Fare (fee paid for ticket)",
     xlab=NULL,
     col="darkgreen"
     )

barplot(table(train.data$Embarked),
        names.arg=c("Cherbourg", "Queenstown", "Southampton"),
        main="Embarked (port of embarkation)", col="sienna"
        )

# mosaic plot
mosaicplot(train.data$Pclass ~ train.data$Survived,
           main="survived by pclass",
           shade=F,
           color=T,
           xlab="Pclass",
           ylab="Survived"
           )
mosaicplot(train.data$Sex ~ train.data$Survived,
           main="survived by sex",
           shade=F,
           color=T,
           xlab="Sex",
           ylab="Survived"
          )
mosaicplot(train.data$Embarked ~ train.data$Survived,
           main="survived by embark",
           shade=F,
           color=T,
           xlab="embark",
           ylab="survived"
)

mosaicplot(train.data$Embarked ~ train.data$Pclass,
           main="pclass by embark",
           shade=F,
           color=T,
           xlab="embark",
           ylab="pclass"
           )

# boxplot
boxplot(train.data$Age ~ train.data$Survived,
        main="Surived by Age",
        xlab="Survived",
        ylab="Age")


boxplot(train.data$Fare ~ train.data$Survived,
        main="Survived by Fare",
        xlab="Survived",
        ylab="Fare")

boxplot(train.data$SibSp ~ train.data$Survived,
        main="Survived by SibSp",
        xlab="Survived",
        ylab="SibSp")

boxplot(train.data$Parch ~ train.data$Survived,
        main="Survived by Parch",
        xlab="Survived",
        ylab="Parch")

boxplot(train.data$Age ~ train.data$Pclass,
        main="Pclass by age",
        xlab="Pclass",
        ylab="Age")

boxplot(train.data$Fare ~ train.data$Pclass,
        main="pclass by Fare",
        xlab="Pclass",
        ylab="Fare")


# corrgram
corrgram.data <- train.data
corrgram.data$Embarked <- revalue(corrgram.data$Embarked,
                                  c("C"=1, "Q"=2, "S"=3))

corrgram.vars <- c("Survived", "Pclass", "Sex", "Age",
                   "SibSp", "Parch", "Fare", "Embarked")


corrgram(corrgram.data[, corrgram.vars],
         order=F,
         lower.panel=panel.shade,
         upper.panel=panel.pie,
         text.panel=panel.txt,
         main="Titanic training data")

# imputation ages
# extract all titles from names

extract.titles <- function(x) {
    m <- str_match(x, "^.+,\\s(.+?)\\..*$")
    return(m[2])
}

train.data$Title <- unlist(lapply(train.data$Name, extract.titles))

#bystats(train.data$Age, train.data$Title,
#        fun=function(x) {c(Mean=mean(x), Median=median(x))})

age.median.data <- ddply(train.data,
                         "Title",
                         summarise,
                         valid.num=sum(!is.na(Age)),
                         NA.num=sum(is.na(Age)),
                         mean=mean(Age, na.rm=T),
                         median=median(Age, na.rm=T))

# impute with median value
impute.titles <- subset(age.median.data, NA.num!=0)$Title
for(t in impute.titles) {
    train.data[train.data$Title==t & is.na(train.data$Age), ]$Age <- subset(age.median.data, Title==t)$median
}


# impute Embarked missing value

train.data$Embarked <- as.factor(train.data$Embarked)
# shown there are two missing value
summary(train.data$Embarked)

# impute with most common value S
train.data$Embarked[which(is.na(train.data$Embarked))] <- 'S'




# explore  age by title
train.data$Title <- factor(train.data$Title,
                           c("Capt","Col","Major","Sir",
                             "Lady","Rev","Dr","Don","Jonkheer",
                             "the Countess","Mrs","Ms","Mr",
                             "Mme","Mlle","Miss","Master"))

boxplot(train.data$Age ~ train.data$Title,
        main="age by title",
        xlab="Title", ylab="Age")

# assign titles with new design ----- feature engineering

train.data$Title[which(train.data$Title=="the Countess")] <- "Mrs"
train.data$Title[which(train.data$Title=="Ms")] <- "Mrs"

train.data$Title[which(train.data$Title=="Mlle")] <- "Miss"
train.data$Title[which(train.data$Title=="Mme")] <- "Miss"

train.data$Title[which(train.data$Title %in%
                       c("Capt", "Col", "Don", "Dr",
                         "Jonkheer", "Lady", "Major",
                         "Rev", "Sir"))] <- "Noble"

train.data$Title <- as.factor(train.data$Title)


# feature
train.data$Survived <- as.factor(train.data$Survived)
train.data$Pclass <- as.factor(train.data$Pclass)
train.data$Sex <- as.factor(train.data$Sex)

featureEngr <- function(df) {
    df$Fate <- df$Survived

    df$Fate <- revalue(df$Fate, c("1"="Survived", "0"="Perished"))

    df$Boat.dibs <- "No"
    df$Boat.dibs[which(df$Sex=="female" | df$Age < 15)] <- "Yes"
    df$Boat.dibs <- as.factor(df$Boat.dibs)

    df$Family <- df$Parch + df$SibSp

    df$Fare.pp <- df$Fare/(df$Family + 1)

    df$Class <- df$Pclass
    df$Class <- revalue(df$Class,
                        c("1"="First", "2"="Second", "3"="Third"))

    df$Deck <- substring(df$Cabin, 1, 1)
    df$Deck[which(is.na(df$Deck))] <- "UNK"
    df$Deck <- as.factor(df$Deck)


    df$cabin.last.digit <- str_sub(df$Cabin, -1)
    df$Side <- "UNK"

    df$Side[which(df$cabin.last.digit %in% c(2, 4, 6, 8, 0))] <- "port"
    df$Side[which(df$cabin.last.digit %in% c(1, 3, 5, 7, 9))] <- "starboard"
    df$Side <- as.factor(df$Side)
    df$cabin.last.digit <- NULL
    return(df)
}

train.data <- featureEngr(train.data)

col.keeps <- c("Fate", "Sex", "Boat.dibs", "Age", "Title",
               "Class", "Deck", "Side", "Fare", "Fare.pp",
               "Embarked", "Family")

train.data.munged <- train.data[col.keeps]


# build a model
set.seed(23)

train.rows <- createDataPartition(train.data.munged$Fate,
                                  p = 0.8, list = F)

# 80% for train, 20% for test
train.batch <- train.data.munged[train.rows, ]
test.batch <- train.data.munged[-train.rows, ]

# using logistic regression
Titanic.logit.1 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare, data = train.batch, family=binomial("logit"))

1-pchisq((951-630), df=8)

anova(Titanic.logit.1, test="Chisq")

# try Fare.pp , but no help
Titanic.logit.2 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare.pp, data=train.batch, family=binomial("logit"))
anova(Titanic.logit.2)

# drop Fare & re-fit as benchmark
Titanic.logit.3 <- glm(Fate ~ Sex + Class + Age + Family + Embarked, data=train.batch, family=binomial("logit"))
anova(Titanic.logit.3)

# train control
# cross validation(CV)

cv.ctrl <- trainControl(method="repeatedcv", repeats=3,
                        summaryFunction=twoClassSummary,
                        classProbs=T)


set.seed(35)

glm.tune.1 <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                    data=train.batch,
                    method="glm",
                    metric="ROC",
                    trControl=cv.ctrl)


glm.tune.2 <- train(Fate ~ Sex + Class + Age + Family + I(Embarked=="S"),
                    data=train.batch, method="glm",
                    metric="ROC", trControl=cv.ctrl)


glm.tune.3 <- train(Fate ~ Sex + Class + Title + Age + Family + I(Embarked=="S"),
                    data=train.batch,
                    method="glm",
                    metric="ROC",
                    trControl=cv.ctrl)

summary(glm.tune.3)



glm.tune.4 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble")
                    + Age + Family + I(Embarked=="S"),
                    data=train.batch,
                    method="glm",
                    metric="ROC",
                    trControl=cv.ctrl)

summary(glm.tune.4)


glm.tune.5 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") + Age + Family + I(Embarked=="S") + I(Title=="Mr" & Class=="Third"),
                    data=train.batch,
                    method="glm",
                    metric="ROC",
                    trControl=cv.ctrl)

summary(glm.tune.5)


# use ada to boosting
# use grid compute to search optimal
ada.grid <- expand.grid(.iter=c(50, 100),
                        .maxdepth=c(4,8),
                        .nu=c(0.1, 1))
set.seed(35)

## ada.tune <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") + Age + Family + I(Embarked=="S") + I(Title=="Mr" & Class=="Third"),
ada.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                  data=train.batch,
                  method="ada",
                  metric="ROC",
                  tuneGrid=ada.grid,
                  trControl=cv.ctrl)

plot(ada.tune)

# random forest
# sole parameter in train is mtry
# mtry : the number of randomly pre-selected predictor variables for each node
# i.e. set mtry at the square root of the number of variables
rf.grid <- data.frame(.mtry=c(2, 3))
rf.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                 data = train.batch,
                 method="rf",
                 metric="ROC",
                 tuneGrid=rf.grid,
                 trControl=cv.ctrl)


# SVM
svm.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                  data=train.batch,
                  method="svmRadial",
                  tuneLength=9,
                  preProcess=c("center", "scale"),
                  metric="ROC",
                  trControl=cv.ctrl)

# model Evaluation

# LR
glm.pred <- predict(glm.tune.5, test.batch)
confusionMatrix(glm.pred, test.batch$Fate)

# Ada
ada.pred <- predict(ada.tune, test.batch)
confusionMatrix(ada.pred, test.batch$Fate)


# RF
rf.pred <- predict(rf.tune, test.batch)
confusionMatrix(rf.pred, test.batch$Fate)

# svm
svm.pred <- predict(svm.tune, test.batch)
confusionMatrix(svm.pred, test.batch$Fate)


# plot ROC curve

glm.probs <- predict(glm.tune.5, test.batch, type="prob")
glm.ROC <- roc(response=test.batch$Fate,
               predictor=glm.probs$Survived,
               levels=levels(test.batch$Fate))
plot(glm.ROC, type="S")

ada.probs <- predict(ada.tune, test.batch, type="prob")
ada.ROC <- roc(response=test.batch$Fate,
               predictor=ada.probs$Survived,
               levels=levels(test.batch$Fate))
plot(ada.ROC, add=T, col="blue")

rf.probs <- predict(rf.tune, test.batch, type="prob")
rf.ROC <- roc(response=test.batch$Fate,
               predictor=rf.probs$Survived,
               levels=levels(test.batch$Fate))
plot(rf.ROC,add=T, col="red")



svm.probs <- predict(svm.tune, test.batch, type="prob")
svm.ROC <- roc(response=test.batch$Fate,
               predictor=svm.probs$Survived,
               levels=levels(test.batch$Fate))
plot(svm.ROC, add=T, col="green")


# perfermance plot

cv.values <- resamples(list(Logit=glm.tune.5, Ada=ada.tune,
                           RF=rf.tune, SVM=svm.tune))

dotplot(cv.values, metrics="ROC")
