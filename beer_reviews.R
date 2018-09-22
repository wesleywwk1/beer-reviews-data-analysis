# read CSV into dataframe
# df <- read.csv("beer_reviews.csv", header=TRUE, sep=",")
# view(df)
name_abv <- list()


beer_name = df$beer_name
beer_abv = df$beer_abv
n = length(beer_abv)

for(i in 1:10){
  
  if(!(beer_name[i] %in% name_abv)){
    
    name_abv[beer_name[i]] = beer_abv[i]  
    
  }
  
}


