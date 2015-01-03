# SPAM/HAM classification using caret and Naive Bayes
Jesus M. Castagnetto  
2015-01-03  



## Background

### Motivation

I am currently reading the book "Machine Learning with R"[^mlr] by Brent Lantz,
and also want to learn more about the `caret`[^caret] package, so I decided to replicate
the SPAM/HAM classification example from the chapter 4 of the book using `caret` 
instead of the `e1071`[^e1071] package used in the text.

Also, instead of using as comparison the number of false positives, I decided
to use the sensitivity and specificity as criteria to evaluate the 
prediction models.

Another difference is that I used the calculated models on a (different) second
dataset to test their prediction performance.

[^mlr]: [Book page at Packt](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-r)

[^caret]: [The caret package](http://caret.r-forge.r-project.org/) site

[^e1071]: [http://cran.r-project.org/web/packages/e1071/index.html](http://cran.r-project.org/web/packages/e1071/index.html)

### Preliminary information

The dataset used in the book is a modified version of the "SMS Spam Collection v.1" created by Tiago A. Almeida and José Maria Gómez Hidalgo[^smsspamcoll],
as described in Chapter 4 ("*Probabilistic Learning -- Clasification Using Naive Bayes") of the aforementioned book. 

[^smsspamcoll]: [http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)

You can get the modified dataset from the book's page at Packt, but be aware
that you will need to register to get the files. If you rather don't do that,
you can get the original data files from the original creator's site.

To simplify things, we are going to use the original dataset.

For this excercise we will use the `caret` package to do the Naive Bayes[^nb]
modeling and prediction, the `tm` package to generate the text corpus,
the `pander` package to be able to output nicely formated tables, 
and the `doMC` to take advantage of parallel processing with multiple cores.
Also, we will define some utility functions to simplify matters later in the code.

[^nb]: The `caret` package uses the `klaR` package for the Naive Bayes algorithm, so we are loading that library and its dependency (`MASS`) beforehand.


```r
# libraries needed by caret
library(klaR)
library(MASS)
# for the Naive Bayes modelling
library(caret)
# to process the text into a corpus
library(tm)
# to get nice looking tables
library(pander)
panderOptions("table.style", "rmarkdown")
# to simplify selections
library(dplyr)
library(doMC)
registerDoMC(cores=4)

# a utility function for % freq tables
frqtab <- function(x, caption) {
    round(100*prop.table(table(x)), 1)
}
# utility function to summarize model comparison results
sumpred <- function(cm) {
    summ <- list(TN=cm$table[1,1],  # true negatives
                 TP=cm$table[2,2],  # true positives
                 FN=cm$table[1,2],  # false negatives
                 FP=cm$table[2,1],  # false positives
                 acc=cm$overall["Accuracy"],  # accuracy
                 sens=cm$byClass["Sensitivity"],  # sensitivity
                 spec=cm$byClass["Specificity"])  # specificity
    lapply(summ, FUN=round, 2)
}
```

## Reading and preparing the data

We start by downloading the zip file with the dataset, and reading the file into
a dataframe. We then assign the appropiate names to the columns, and convert
the `type` into a factor. Finally, we randomize the data frame.


```r
if (!file.exists("smsspamcollection.zip")) {
download.file(url="http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/smsspamcollection.zip",
              destfile="smsspamcollection.zip", method="curl")
}
sms_raw <- read.table(unz("smsspamcollection.zip","SMSSpamCollection"),
                      header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
colnames(sms_raw) <- c("type", "text")
sms_raw$type <- factor(sms_raw$type)
# randomize it a bit
set.seed(12358)
sms_raw <- sms_raw[sample(nrow(sms_raw)),]
str(sms_raw)
```

```
'data.frame':	5574 obs. of  2 variables:
 $ type: Factor w/ 2 levels "ham","spam": 1 1 1 1 1 1 2 1 1 1 ...
 $ text: chr  "Honeybee Said: *I'm d Sweetest in d World* God Laughed &amp; Said: *Wait,U Havnt Met d Person Reading This Msg* MORAL: Even GOD"| __truncated__ "Ha ha ha good joke. Girls are situation seekers." "You sure your neighbors didnt pick it up" ", im .. On the snowboarding trip. I was wondering if your planning to get everyone together befor we go..a meet and greet kind "| __truncated__ ...
```

The modified data used in the book has 5559 SMS messages, whereas the original
data used here has 5574 rows (*caveat*: I have not checked for
duplicates in the original dataset).

## Preparing the data

We wil proceed in a similar fashion as described in the book, but make use
of `dplyr` syntax to execute the text cleanup/transformation operations

First we will transform the SMS text into a corpus that can later be used in the
analysis, then we will convert all text to lowercase, remove numbers, remove 
some common *stop words* in english, remove punctuation and extra whitespace,
and finally, generate the document term that will be the basis for the 
classification task.


```r
sms_corpus <- Corpus(VectorSource(sms_raw$text))
sms_corpus_clean <- sms_corpus %>%
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind="en")) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
```

## Creating a classification model witn Naive Bayes

### Generating the training and testing datasets

We will use the `createDataPartition` function to split the original dataset
into a training and a testing sets, using the proportions from the book (75% 
training, 25% testing). This also generates the corresponding corpora and
document term matrices.

According to the documentation that accompanies the data file, 86.6% of the
entries correspond to legitimate messages ("ham"), and 13.4% to spam messages.
We shall see if the partition procedure has preserved those proportions in the
testing and training sets.


```r
train_index <- createDataPartition(sms_raw$type, p=0.75, list=FALSE)
sms_raw_train <- sms_raw[train_index,]
sms_raw_test <- sms_raw[-train_index,]
sms_corpus_clean_train <- sms_corpus_clean[train_index]
sms_corpus_clean_test <- sms_corpus_clean[-train_index]
sms_dtm_train <- sms_dtm[train_index,]
sms_dtm_test <- sms_dtm[-train_index,]

ft_orig <- frqtab(sms_raw$type)
ft_train <- frqtab(sms_raw_train$type)
ft_test <- frqtab(sms_raw_test$type)
ft_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ft_df) <- c("Original", "Training set", "Test set")
pander(ft_df,
       caption=paste0("Comparison of SMS type frequencies among datasets"))
```



|   &nbsp;   |  Original  |  Training set  |  Test set  |
|:----------:|:----------:|:--------------:|:----------:|
|  **ham**   |    86.6    |      86.6      |    86.6    |
|  **spam**  |    13.4    |      13.4      |    13.4    |

Table: Comparison of SMS type frequencies among datasets

It would seem that the procedure keeps the proportions perfectly.

Following the strategy used in the book, we will pick terms that appear at least
5 times in the training document term matrix. To do this, we first create a 
dictionary of terms (using the function `findFreqTerms`) that we will use to
filter the cleaned up training and testing corpora. 

As a final step before using these sets, we will convert the numeric entries in
the term matrices into factors that indicate whether the term is present or not.
For this, we'll use a slightly modified version of the `convert_counts` function
that appear in the book, and apply it to each column in the matrices.


```r
sms_dict <- findFreqTerms(sms_dtm_train, lowfreq=5)
sms_train <- DocumentTermMatrix(sms_corpus_clean_train, list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_clean_test, list(dictionary=sms_dict))

# modified sligtly fron the code in the book
convert_counts <- function(x) {
    x <- ifelse(x > 0, 1, 0)
    x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}
sms_train <- sms_train %>% apply(MARGIN=2, FUN=convert_counts)
sms_test <- sms_test %>% apply(MARGIN=2, FUN=convert_counts)
```

### Training the two prediction models

We will now use Naive Bayes to train a couple of prediction models.
Both models will be generated using 10-fold cross validation, with the
default parameters.

The difference between the models will be that the first one does not
use the Laplace correction and let's the training procedure figure out whether
to user or not a kernel density estimate, while the second one fixes Laplace
parameter to one (`fL=1`) and explicitly forbids the use of a kernel density 
estimate (`useKernel=FALSE`). 


```r
ctrl <- trainControl(method="cv", 10)
set.seed(12358)
sms_model1 <- train(sms_train, sms_raw_train$type, method="nb",
                trControl=ctrl)
sms_model1
```

```
Naive Bayes 

4182 samples
1203 predictors
   2 classes: 'ham', 'spam' 

No pre-processing
Resampling: Cross-Validated (10 fold) 

Summary of sample sizes: 3764, 3764, 3764, 3763, 3764, 3764, ... 

Resampling results across tuning parameters:

  usekernel  Accuracy   Kappa      Accuracy SD  Kappa SD  
  FALSE      0.9811085  0.9143632  0.004139905  0.02047767
   TRUE      0.9811085  0.9143632  0.004139905  0.02047767

Tuning parameter 'fL' was held constant at a value of 0
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were fL = 0 and usekernel = FALSE. 
```

```r
set.seed(12358)
sms_model2 <- train(sms_train, sms_raw_train$type, method="nb", 
                    tuneGrid=data.frame(.fL=1, .usekernel=FALSE),
                trControl=ctrl)
sms_model2
```

```
Naive Bayes 

4182 samples
1203 predictors
   2 classes: 'ham', 'spam' 

No pre-processing
Resampling: Cross-Validated (10 fold) 

Summary of sample sizes: 3764, 3764, 3764, 3763, 3764, 3764, ... 

Resampling results

  Accuracy   Kappa      Accuracy SD  Kappa SD 
  0.9808698  0.9125901  0.005752356  0.0284413

Tuning parameter 'fL' was held constant at a value of 1
Tuning
 parameter 'usekernel' was held constant at a value of FALSE
 
```

### Testing the predictions

We now use these two models to predict the appropriate classification of the
terms in the test set. In each case we will estimate how good is the prediction
using the `confusionMatrix` function. We will consider a positive result when
a message is identified as (or predicted to be) SPAM.


```r
sms_predict1 <- predict(sms_model1, sms_test)
cm1 <- confusionMatrix(sms_predict1, sms_raw_test$type, positive="spam")
cm1
```

```
Confusion Matrix and Statistics

          Reference
Prediction  ham spam
      ham  1199   23
      spam    7  163
                                          
               Accuracy : 0.9784          
                 95% CI : (0.9694, 0.9854)
    No Information Rate : 0.8664          
    P-Value [Acc > NIR] : < 2e-16         
                                          
                  Kappa : 0.9034          
 Mcnemar's Test P-Value : 0.00617         
                                          
            Sensitivity : 0.8763          
            Specificity : 0.9942          
         Pos Pred Value : 0.9588          
         Neg Pred Value : 0.9812          
             Prevalence : 0.1336          
         Detection Rate : 0.1171          
   Detection Prevalence : 0.1221          
      Balanced Accuracy : 0.9353          
                                          
       'Positive' Class : spam            
                                          
```

```r
sms_predict2 <- predict(sms_model2, sms_test)
cm2 <- confusionMatrix(sms_predict2, sms_raw_test$type, positive="spam")
cm2
```

```
Confusion Matrix and Statistics

          Reference
Prediction  ham spam
      ham  1203   30
      spam    3  156
                                          
               Accuracy : 0.9763          
                 95% CI : (0.9669, 0.9836)
    No Information Rate : 0.8664          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.8909          
 Mcnemar's Test P-Value : 6.011e-06       
                                          
            Sensitivity : 0.8387          
            Specificity : 0.9975          
         Pos Pred Value : 0.9811          
         Neg Pred Value : 0.9757          
             Prevalence : 0.1336          
         Detection Rate : 0.1121          
   Detection Prevalence : 0.1142          
      Balanced Accuracy : 0.9181          
                                          
       'Positive' Class : spam            
                                          
```

We will also use our `sumpred` function to extract the *true positives* (TP), 
*true negatives* (TN), *false positives* (FP), and *false negatives* (FN), 
the prediction *accuracy*[^acc], the *sensitivity*[^sens] (also
known as *recall* or *true positive rate*), and the *specificity*[^spec] 
(also known as *true negative rate*).

Also, we will use the information from the similar models described in the book,
in terms of TP, TN, TP, and FN, to estimate the rest of the parameters, and
compare them with the `caret` derived models.

[^acc]: Accuracy, is the degree of closeness of measurements of a quantity to that quantity's actual (true) value.

[^sens]: The sensitivity, measures the proportion of actual positives which are correctly identified as such.

[^spec]: The specificity,measures the proportion of negatives which are correctly identified as such.


```r
# from the table on page 115 of the book
tn=1203
tp=151
fn=32
fp=4
book_example1 <- list(
    TN=tn,
    TP=tp,
    FN=fn,
    FP=fp,
    acc=(tp + tn)/(tp + tn + fp + fn),
    sens=tp/(tp + fn),
    spec=tn/(tn + fp))

# from the table on page 116 of the book
tn=1204
tp=152
fn=31
fp=3
book_example2 <- list(
    TN=tn,
    TP=tp,
    FN=fn,
    FP=fp,
    acc=(tp + tn)/(tp + tn + fp + fn),
    sens=tp/(tp + fn),
    spec=tn/(tn + fp))

b1 <- lapply(book_example1, FUN=round, 2)
b2 <- lapply(book_example2, FUN=round, 2)
m1 <- sumpred(cm1)
m2 <- sumpred(cm2)
model_comp <- as.data.frame(rbind(b1, b2, m1, m2))
rownames(model_comp) <- c("Book model 1", "Book model 2", "Caret model 1", "Caret model 2")
pander(model_comp, split.tables=Inf, keep.trailing.zeros=TRUE,
       caption="Model results when comparing predictions and test set")
```



|       &nbsp;        |  TN  |  TP  |  FN  |  FP  |  acc  |  sens  |  spec  |
|:-------------------:|:----:|:----:|:----:|:----:|:-----:|:------:|:------:|
|  **Book model 1**   | 1203 | 151  |  32  |  4   | 0.97  |  0.83  |   1    |
|  **Book model 2**   | 1204 | 152  |  31  |  3   | 0.98  |  0.83  |   1    |
|  **Caret model 1**  | 1199 | 163  |  23  |  7   | 0.98  |  0.88  |  0.99  |
|  **Caret model 2**  | 1203 | 156  |  30  |  3   | 0.98  |  0.84  |   1    |

Table: Model results when comparing predictions and test set

Accuracy gives us an overall sense of how good the models are, and
using that criteria, the ones in the book and those calculated here are very
similar in how well they classify an SMS. All of them do surprisingly well 
taking into account the simplicity of the method.

The discussion in the book centered around the number of FP predicted by the
model, but I'd rather look at the sensitivity (related to type II errors) and 
specificity (related to Type I errors) of the predictions (and the corresponding 
PPV and NPV).

In this example, the sensitivity gives us the probability of an SMS
text being classified as SPAM, when it really is SPAM. Looking at this 
parameter, we see that even though the book's models do not differ much
from the `caret` models  in terms of accuracy, they do worse in terms of
sensitivity. The text of the book argues that using the Laplace correction
improves prediction, but with the cross-validated models generated using
`caret` the opposite is true.

Of course, we gain in sensitivity, but we lose slightly in specificity, which in
this example is the probability of a HAM message being classified as HAM. In
other words, we increase (a bit) the misclassification of the regular SMS texts
as SPAM. But the difference between the worst and the best specificity is of the
order of 0.01 or ~1%.

## Applying the model to a different SMS SPAM dataset

Just to check if our `caret` Naive Bayes models are good enough, we will test
them against a different corpus. One described "Independent and Personal SMS Spam Filtering."[^brit_spam]

This dataset can be obtained from one of the authors site[^brit_spam_data], 
as the `british-english-sms-corpora.doc` MSWORD document (retrieved on 
2014-12-30). This document was converted to a text file using the 
Unix/Linux `catdoc` command (`$ catdoc -aw british-english-sms-corpora.doc > british-english-sms-corpora.txt`), and then split into two
archives: one containing all the "ham" messages 
(`british-english-sms-corpora_ham.txt`), and the other containing all the
"spam" message (`british-english-sms-corpora_spam.txt`).

[^brit_spam]: "Independent and Personal SMS Spam Filtering." in Proc. of IEEE Conference on Computer and Information Technology, Paphos, Cyprus, Aug 2011. Page 429-435. (URL: [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?reload=true&arnumber=6036805](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?reload=true&arnumber=6036805))

[^brit_spam_data]: http://mtaufiqnzz.wordpress.com/british-english-sms-corpora/

These two archive were then read and combined into a data frame. The data was
randomized to get a suitable dataset for the rest of the procedures.

The proportion of the spam and ham messages can be seen in the following table.


```r
brit_ham <- data.frame(
    type="ham",
    text=readLines("british-english-sms-corpora_ham.txt"),
    stringsAsFactors=FALSE)
brit_spam <- data.frame(
    type="spam",
    text=readLines("british-english-sms-corpora_spam.txt"),
    stringsAsFactors=FALSE)
brit_sms <- data.frame(rbind(brit_ham, brit_spam))
brit_sms$type <- factor(brit_sms$type)
set.seed(12358)
brit_sms <- brit_sms[sample(nrow(brit_sms)),]
pander(frqtab(brit_sms$type), caption="Proportions in the new SMS dataset")
```


------------
 ham   spam 
----- ------
51.4   48.6 
------------

Table: Proportions in the new SMS dataset

As before, we convert the text into a corpus, clean it up, and generate a
filtered document term matrix. The term counts in the matrix are converted
into factors, and we generate predictions using both `caret` models.

Also, we calculate the confusion matrix for both predictions.


```r
brit_corpus <- Corpus(VectorSource(brit_sms$text))
brit_corpus_clean <- brit_corpus %>%
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords()) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
brit_dtm <- DocumentTermMatrix(brit_corpus_clean, list(dictionary=sms_dict))
brit_test <- brit_dtm %>% apply(MARGIN=2, FUN=convert_counts)
brit_predict1 <- predict(sms_model1, brit_test)
brit_cm1 <- confusionMatrix(brit_predict1, brit_sms$type, positive="spam")
brit_cm1
```

```
Confusion Matrix and Statistics

          Reference
Prediction ham spam
      ham  449   60
      spam   1  365
                                          
               Accuracy : 0.9303          
                 95% CI : (0.9113, 0.9463)
    No Information Rate : 0.5143          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.8599          
 Mcnemar's Test P-Value : 1.118e-13       
                                          
            Sensitivity : 0.8588          
            Specificity : 0.9978          
         Pos Pred Value : 0.9973          
         Neg Pred Value : 0.8821          
             Prevalence : 0.4857          
         Detection Rate : 0.4171          
   Detection Prevalence : 0.4183          
      Balanced Accuracy : 0.9283          
                                          
       'Positive' Class : spam            
                                          
```

```r
brit_predict2 <- predict(sms_model2, brit_test)
brit_cm2 <- confusionMatrix(brit_predict2, brit_sms$type, positive="spam")
brit_cm2
```

```
Confusion Matrix and Statistics

          Reference
Prediction ham spam
      ham  450   72
      spam   0  353
                                          
               Accuracy : 0.9177          
                 95% CI : (0.8975, 0.9351)
    No Information Rate : 0.5143          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.8345          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8306          
            Specificity : 1.0000          
         Pos Pred Value : 1.0000          
         Neg Pred Value : 0.8621          
             Prevalence : 0.4857          
         Detection Rate : 0.4034          
   Detection Prevalence : 0.4034          
      Balanced Accuracy : 0.9153          
                                          
       'Positive' Class : spam            
                                          
```

When comparing the predictions on this new dataset, it is suprising that we
still have a very good accuracy and sensitivity. Even though the Naive Bayes
approach is simplistic, and makes some assumptions that are not always correct,
it tends to do well with SPAM classification, which is the reason why is so
widely used for this purpose.


```r
bm1 <- sumpred(brit_cm1)
bm2 <- sumpred(brit_cm2)
model_comp <- as.data.frame(rbind(bm1,bm2))
rownames(model_comp) <- c("Caret model 1", "Caret model 2")
pander(model_comp, split.tables=Inf, keep.trailing.zeros=TRUE,
       caption="Applying the caret models to a new dataset")
```


-----------------------------------------------------------
      &nbsp;         TN   TP   FN   FP   acc   sens   spec 
------------------- ---- ---- ---- ---- ----- ------ ------
 **Caret model 1**  449  365   60   1   0.93   0.86    1   

 **Caret model 2**  450  353   72   0   0.92   0.83    1   
-----------------------------------------------------------

Table: Applying the caret models to a new dataset

## A cautionary tale about/against self-deception

Initially, when looking for a second dataset to contrast the models, I found
"The SMS Spam Corpus v.0.1 Big"[^smsspamcorpus]
by José María Gómez Hidalgo and Enrique Puertas Sanz, which contains a
total of 1002 legitimate messages and a total of 322 spam messages.

When I redid the calculations on this dataset, I found a extraordinarily good
prediction rate (*vide infra*), and initially I was amazed at how well the
model performed, how well Naive Bayes can perform this textual classification, 
etc. That is, until I decided to check if there was any overlap
between this dataset and the first one we used: Out of 1324 rows, 1274 were
the same between both datasets. I later found out that the dataset I just
obtained was a previous version of the first one -- something I should've 
noticed before doing the analysis.

[^smsspamcorpus]: [http://www.esp.uem.es/jmgomez/smsspamcorpus](http://www.esp.uem.es/jmgomez/smsspamcorpus)

Just out of completeness I will describe the procedure used to process this dataset. The data file of the "The SMS Spam Corpus v.0.1 Big" has inconsistent
formatting, so we need to clean it up a bit before using it in our analysis.


```r
library(plyr)
smsdata <- readLines(unz("SMSSpamCorpus01.zip", "english_big.txt")) %>%
    iconv(from="latin1", to="UTF-8")
# make it easy to parse later
smsdata <- gsub(",(spam|ham)$", "|\\1", smsdata)
smsdata <- smsdata %>% strsplit("|", fixed=TRUE) %>% ldply()
colnames(smsdata) <- c("text", "type")
smsdata$type <- factor(smsdata$type)
# generate the corpus
sms_corpus2 <- Corpus(VectorSource(smsdata$text))
sms_corpus2_clean <- sms_corpus2 %>%
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords()) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
# and the dtm
sms_dtm2 <- DocumentTermMatrix(sms_corpus2_clean, list(dictionary=sms_dict))
sms_test2 <- sms_dtm2 %>% apply(MARGIN=2, FUN=convert_counts)

# do the predictions using the caret models
sms_predict3 <- predict(sms_model1, sms_test2)
cm3 <- confusionMatrix(sms_predict3, smsdata$type, positive="spam")
cm3
```

```
Confusion Matrix and Statistics

          Reference
Prediction  ham spam
      ham  1001   22
      spam    1  299
                                         
               Accuracy : 0.9826         
                 95% CI : (0.974, 0.9889)
    No Information Rate : 0.7574         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9516         
 Mcnemar's Test P-Value : 3.042e-05      
                                         
            Sensitivity : 0.9315         
            Specificity : 0.9990         
         Pos Pred Value : 0.9967         
         Neg Pred Value : 0.9785         
             Prevalence : 0.2426         
         Detection Rate : 0.2260         
   Detection Prevalence : 0.2268         
      Balanced Accuracy : 0.9652         
                                         
       'Positive' Class : spam           
                                         
```

```r
sms_predict4 <- predict(sms_model2, sms_test2)
cm4 <- confusionMatrix(sms_predict4, smsdata$type, positive="spam")
cm4
```

```
Confusion Matrix and Statistics

          Reference
Prediction  ham spam
      ham  1002   23
      spam    0  298
                                         
               Accuracy : 0.9826         
                 95% CI : (0.974, 0.9889)
    No Information Rate : 0.7574         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9515         
 Mcnemar's Test P-Value : 4.49e-06       
                                         
            Sensitivity : 0.9283         
            Specificity : 1.0000         
         Pos Pred Value : 1.0000         
         Neg Pred Value : 0.9776         
             Prevalence : 0.2426         
         Detection Rate : 0.2252         
   Detection Prevalence : 0.2252         
      Balanced Accuracy : 0.9642         
                                         
       'Positive' Class : spam           
                                         
```

As we can see below, the two models fit the dataset like an old glove. And if
you read the explanation above, it makes sense that this happens.


```r
m3 <- sumpred(cm3)
m4 <- sumpred(cm4)
model_comp <- as.data.frame(rbind(m3,m4))
rownames(model_comp) <- c("Caret model 1", "Caret model 2")
pander(model_comp, split.tables=Inf, keep.trailing.zeros=TRUE,
       caption="Applying the caret models to a new dataset")
```


-----------------------------------------------------------
      &nbsp;         TN   TP   FN   FP   acc   sens   spec 
------------------- ---- ---- ---- ---- ----- ------ ------
 **Caret model 1**  1001 299   22   1   0.98   0.93    1   

 **Caret model 2**  1002 298   23   0   0.98   0.93    1   
-----------------------------------------------------------

Table: Applying the caret models to a new dataset

## Reproducibility information

The dataset used to generate the models is the original "SMS Spam Collection v.1"
by Tiago A. Almeida and José Maria Gómez Hidalgo[^smsspamcoll], 
as described in the chapter 4 of book "Machine Learning with R" by Brett Lantz 
(ISBN 978-1-78216-214-8).

The dataset used to contrast the models is the "British English SMS Corpora",
obtained on 2014-12-30 from one of the dataset author's [http://mtaufiqnzz.wordpress.com/british-english-sms-corpora/](http://mtaufiqnzz.wordpress.com/british-english-sms-corpora/).


```r
sessionInfo()
```

```
R version 3.1.2 (2014-10-31)
Platform: x86_64-pc-linux-gnu (64-bit)

locale:
 [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
 [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
 [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
 [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
 [9] LC_ADDRESS=C               LC_TELEPHONE=C            
[11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       

attached base packages:
[1] parallel  stats     graphics  grDevices utils     datasets  methods  
[8] base     

other attached packages:
 [1] plyr_1.8.1      klaR_0.6-12     MASS_7.3-33     doMC_1.3.3     
 [5] iterators_1.0.7 foreach_1.4.2   dplyr_0.2       pander_0.3.8   
 [9] tm_0.6          NLP_0.1-5       caret_6.0-37    ggplot2_1.0.0  
[13] lattice_0.20-29 knitr_1.8      

loaded via a namespace (and not attached):
 [1] assertthat_0.1      BradleyTerry2_1.0-5 brglm_0.5-9        
 [4] car_2.0-19          class_7.3-10        codetools_0.2-9    
 [7] colorspace_1.2-2    combinat_0.0-8      compiler_3.1.2     
[10] digest_0.6.4        e1071_1.6-3         evaluate_0.5.5     
[13] formatR_1.0         grid_3.1.2          gtable_0.1.2       
[16] gtools_3.4.1        htmltools_0.2.6     lme4_1.1-6         
[19] magrittr_1.0.1      Matrix_1.1-4        minqa_1.2.3        
[22] munsell_0.4.2       nlme_3.1-118        nnet_7.3-8         
[25] proto_0.3-10        Rcpp_0.11.2         RcppEigen_0.3.2.1.2
[28] reshape2_1.4        rmarkdown_0.2.64    scales_0.2.4       
[31] slam_0.1-32         splines_3.1.2       stringr_0.6.2      
[34] tools_3.1.2         yaml_2.1.13        
```
