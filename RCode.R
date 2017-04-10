###### Example 4: Airlines Network using Edge List ##################
dir()

airports <- read.csv("E:/Data Mining/Assignment 2/airports.dat", header=FALSE) ## source: http://openflights.org/data.html
colnames(airports) <- c("Airport ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","Tz database timezone")
head(airports)

airline_routes <- read.csv("E:/Data Mining/Assignment 2/routes.dat", header=FALSE) ## source: http://openflights.org/data.html
colnames(airline_routes) <- c("Airline", "Airline ID", "Source Airport","Source Airport ID","Destination Airport","Destination Airport ID","Codeshare","Stops","Equipment")
head(airline_routes)


AirlineNW <- graph.edgelist(as.matrix(airline_routes[,c(3,5)]), directed=TRUE)
plot(AirlineNW)

AAA <- leading.eigenvector.community(AirlineNW)
#membership(AAA)
plot(AAA,AirlineNW)

#How many distinct airports are there in the dataset? How many communities of airports got identified? List the number of airports in each cluster/community in a table.
length(unique(AAA$names))

#Step 5: Compute the centralities (in-degree, out-degree, in-closeness, eigenvector, betweenness) of each airport. Now, run k-Means clustering to group the airports based on their centralities alone. Take k as the number of communities you obtained in Part c, above.
# table of communities and their counts
comm <- membership(AAA)
table(comm)

#Indegree
indegree <- degree(AirlineNW,mode="in")
max(indegree)

head(indegree)

#OutDegree

outdegree <- degree(AirlineNW,mode="out")
max(outdegree)
head(outdegree)

#INcloseness
closeness_in <- closeness(AirlineNW, mode="in",normalized = TRUE)
max(closeness_in)
head(closeness_in)

#Eigen Vector
indegree <- degree(AirlineNW,mode="in")
max(indegree)
head(indegree)

pg_rank <- page_rank(AirlineNW,damping = 0)
pg_rank$vector

max(pg_rank$vector)

## K Means Clustering
setwd("E:/Data Mining/Assignment 2")
airports <- read.csv("airports.csv",header=FALSE) ## source: http://openflights.org/data.html
colnames(airports) <- c("Airport ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","Tz database timezone")

setwd("E:/Data Mining/Assignment 2")
airline_routes <- read.csv("routes.csv",header=FALSE) ## source: http://openflights.org/data.html
colnames(airline_routes) <- c("Airline", "Airline ID", "Source Airport","Source Airport ID","Destination Airport","Destination Airport ID","Codeshare","Stops","Equipment")

AirlineNW <- graph.edgelist(as.matrix(airline_routes[,c(3,5)]),directed=TRUE)

indegree <- degree(AirlineNW,mode="in")
outdegree <- degree(AirlineNW,mode="out")
closeness_in <- closeness(AirlineNW, mode="in",normalized = TRUE)
btwn <- betweenness(AirlineNW,normalized = TRUE)

centralities <- cbind(indegree,outdegree,closeness_in,btwn)

normalized_data <- scale(centralities)

fit <- kmeans(normalized_data, 25) # 3 cluster solution

fit

#Difference
sum(fit$cluster==membership(AAA)) 

#(e) Identify a few (3-4) airports that you already know of to some extent in terms of their characteristics. Explain how centralities of those airports relative to that of the other airports in the network are in line with the characteristics you already know of about those airports.
centralities<- data.frame(centralities, fit$cluster) # append cluster membership
head(centralities)

aggregate(centralities, by=list(fit$cluster), FUN=mean) 




head(centralities)


