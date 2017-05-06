#Load Data

library(data.table)
business <- fread("yelp_academic_dataset_business_train.csv")
#business$stars <- as.factor(business$stars)
business_test <- fread("yelp_academic_dataset_business_test.csv")
#checkin <- fread("yelp_academic_dataset_checkin.csv")
#tips <- fread("yelp_academic_dataset_tip.csv")
review_train <- fread("yelp_academic_dataset_review_train.csv")
review_test <- fread("yelp_academic_dataset_review_test.csv")

#NLP - Raj

#clean train data
library(tm)
library(nnet)
library(e1071)
library(data.table)

review_train = fread('yelp_academic_dataset_review_train.csv')
reviews = as.vector(review_train$text)
#stopWords = c(stopwords("en"), "", "cant", "dont", "got", "around", "one", "anyway", "bit", "since", "maybe", "ive", "mine", "theres", "las", "say", "youll") 
stopWords = c(stopwords("en"), "")

#need to keep apostrophe marks - in stop words list
cleanReview = function(review, stop_words=stopWords){
  # In your project, you could modify this function 
  # to modify a review however you'd like (e.g. 
  # add more stop words or spell checker -
  # This is a VERY preliminary version 
  # of this function)
  
  # Lowercase review 
  lower_txt = tolower(review)
  # Remove punctuation - (might want to keep !)
  lower_txt = gsub("[[:punct:]]", " ", lower_txt)
  # Tokenize review 
  tokens = strsplit(lower_txt, ' ')[[1]]
  # Remove stop words 
  clean_tokens = tokens[!(tokens %in% stopWords)]
  clean_review = paste(clean_tokens, collapse=' ')
  return(clean_review)
}

cleanCorpus = function(corpus){
  # You can also use this function instead of the first. 
  # Here you clean all the reviews at once using the 
  # 'tm' package. Again, a lot more you can add to this function...
  
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

clean_reviews = sapply(reviews, function(review) cleanReview(review))
review_corpus = Corpus(VectorSource(clean_reviews))

# Create document term matrix - try TD-IDF
review_dtm = DocumentTermMatrix(review_corpus) # 93k terms

# Remove less frequent words
review_dtm = removeSparseTerms(review_dtm, 0.99) # 813 terms


# # Training sets
X_train = as.matrix(review_dtm)
X_train = as.data.frame(as.matrix(X_train))
y_train = review_train$stars


#review train_train
train_train <- sample(1:116474, 75708, replace = F)
#X_train_bin <- (X_train > 0) * 1



#clean test data
reviews <- review_test$text
clean_reviews = sapply(reviews, function(review) cleanReview(review))
review_corpus = Corpus(VectorSource(clean_reviews))

# Create document term matrix - try TD-IDF
review_dtm = DocumentTermMatrix(review_corpus) # 93k terms

# Remove less frequent words
review_dtm = removeSparseTerms(review_dtm, 0.99) # 813 terms


X_test = as.matrix(review_dtm)
X_test <- as.data.frame(as.matrix(X_test))

rm(clean_reviews)
rm(review_corpus)
rm(review_dtm)
rm(reviews)

#Attribute Parsing

#for attributes that are pretty much just yes/no
attribution <- function(frame, regexPattern, asFactor = F) {
  attribs <- 0
  for (i in 1:nrow(frame)) {
    attrib <- frame$attributes[i]
    if (grepl(x = attrib, pattern = regexPattern, fixed = T)) {
      attrib_loc <- regexpr(regexPattern, attrib)[1]
      attrib <- substring(attrib, attrib_loc)
      apostrophe_loc <- regexpr("'", attrib)
      attribs[i] <- substring(attrib, nchar(regexPattern) + 3, 
                              apostrophe_loc - 1)
    } else {
      attribs[i] <- "NA"
    }
  }
  if (asFactor) {
    attribs <- as.factor(attribs)
  }
  frame <- cbind(frame, attribs)
  names(frame) <- c(names(frame)[1:(length(names(frame)) - 1)], regexPattern)
  return(frame)
}

attr_names <- c("WiFi", "BusinessAcceptsCreditCards", "OutdoorSeating", "Alcohol", "GoodForKids", "RestaurantsTakeOut", "RestaurantsGoodForGroups", "RestaurantsTableService", "RestaurantsReservations", "RestaurantsDelivery", "RestaurantsAttire", "RestaurantsPriceRange2")


for (i in 1:length(attr_names)) {
  business <- attribution(business, attr_names[i])
  business_test <- attribution(business_test, attr_names[i])
}


#Test on X_test

#Raj linear model
basic_lin_mod = lm(y_train ~ X_train)
mean(abs(basic_lin_mod$residuals)) # training error .7937

raj_model_pred_final <- predict(basic_lin_mod, X_test) #test gives different word space

#attribute linear model
business_train <- business[,c(9, 10, 12, 15, 13, 18:29, 14)]
train_train_lm <- sample(1:2510, 2008, replace = F)
business_lm <- lm(stars ~ ., data = business_train[train_train_lm])
business_lm_pred <- predict(business_lm, business_train[-train_train_lm])
business_lm_pred_rmse <- sqrt(mean((business_lm_pred - business_train$stars)^2)) #.8007
business_lm_pred_test_final <- predict(business_lm, business_test[,c(10, 11, 13, 15, 14, 18:29)])
#business_predictions <- data.frame("business_id" = business_test$business_id, "stars" = business_attrib_lm_pred_test)

#review xgboost model
library(xgboost)
# train_train <- sample(1:116474, 75708, replace = F)
# X_train_bin <- (X_train > 0) * 1
business_xg <- xgboost(data = data.matrix(X_train[train_train,]),
                       label = y_train[train_train],
                       max.depth = 10, nrounds = 40)

#test on train_test data
business_xg_train_test_pred <- predict(business_xg, data.matrix(X_train[-train_train,]))
business_xg_rmse <- sqrt(mean((business_xg_train_test_pred - y_train[-train_train])^2)) #.9434

business_xg_test_pred_final <- predict(business_xg, data.matrix(X_test))

business_id_test <- review_test$business_id
x_test <- data.frame("business_id" = business_id_test, "stars" = business_xg_test_pred_final)
library(dplyr)
#need to reformat as data frame, set names and whatnot
reviews_test_predictions <- x_test %>% group_by(., business_id) %>% summarise(., stars = mean(stars))
business_test_id <- unique(business_test$business_id)
reviews_test_predictions_ordered <- matrix(0, ncol = 2)
for (i in 1:length(business_test_id)) {
  reviews_test_predictions_ordered <- rbind(reviews_test_predictions_ordered, c(business_test_id[i], reviews_test_predictions$stars[which(reviews_test_predictions$business_id == business_test_id[i])]))
}
reviews_test_predictions_ordered <- reviews_test_predictions_ordered[-1,]
reviews_test_predictions_ordered <- as.data.frame(reviews_test_predictions_ordered)
names(reviews_test_predictions_ordered) <- c("business_id", "stars")
reviews_test_predictions_ordered$stars <- as.numeric(as.character(reviews_test_predictions_ordered$stars))

#review cart model
library(rpart)
library(rpart.plot)
X_train_df <- as.data.frame(as.matrix(cbind(X_train[train_train,], y_train[train_train])))
X_train_df <- cbind(X_train_df, y_train[train_train])
names(X_train_df) <- c(names(X_train_df)[1:813], "Review_star")
review_cart <- rpart(Review_star ~ ., data = X_train_df)

review_cart_pred <- predict(review_cart, as.data.frame(as.matrix(X_train[-train_train,])))
review_cart_pred_rmse <- sqrt(mean((review_cart_pred - y_train[-train_train])^2)) #1.2068

review_cart_pred_final <- predict(review_cart, X_test)