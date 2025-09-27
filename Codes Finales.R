library(ggplot2)
library(leaps)
library(forecast)
library(glmnet)

rm(list=objects())
graphics.off()

library(readxl) #Pour la lecture de fichier xlsx
library("FactoMineR")
library(ggplot2)
library(factoextra)
raisin <- read_excel("Raisin.xlsx")
View(Raisin)
n = nrow(raisin)
p = ncol(raisin)
boxplot(raisin[,-p])
#Avec de la couleur
library(reshape2)
data_mod = melt(raisin, id.vars=1,       # variable à recycler en fait a peu d'importance dans notre cas
                measure.vars=1:(p-1)) # variable à empiler
# creating a plot
pl = ggplot(data_mod) +
  geom_boxplot(aes(x=variable, y=value, color=variable))+
  xlab("variables")
pl

# Un boxplot en fonction des niveaux d'un facteur
ggplot(raisin)+
  geom_boxplot(aes(x=as.factor(Class), y=Eccentricity, col=as.factor(Class)))


#Analyse bivariée
pairs(raisin[,-p])
library(corrplot)
corrplot(cor(raisin[,-p]) , method ="circle")
library(GGally)
ggpairs(raisin[,-p])

#Question 2 : ACP
#Étape 1 : centrer et réduire les données
X = scale(raisin[,-p],center=TRUE,scale=TRUE)/sqrt((n-1)/n)

#Étape 2 Calcul des valeurs propres et ébouli des valeurs propres
eigen(cor(X))
sum <- sum(eigen(cor(X))$values)
prop <-eigen(cor(X))$values/sum
barplot(prop)

#Affichage
D=data.frame(eigen(cor(X))$values,prop)
g <- ggplot()+aes(x=1:length(prop),y=prop)+geom_col()+xlab("")
plot(g)

#Les deux premières directions permettent de conserver 80% des valeurs propres,
#on peut aussi appliquer le critère du coude.

#Étape 3 Cercle des corrélations dans le premier plan principal
#cercle des corrélations
#Calcul des composantes principales
X_mat <- as.matrix(X)
F1 <- X_mat%*%eigen(cor(X))$vectors[,1]
F2 <- X_mat%*%eigen(cor(X))$vectors[,2]
G1=cor(X,F1)
G2=cor(X,-F2)
var_var=colnames(X)
circle <- annotate("path", x=0+1*cos(seq(0,2*pi,length.out=100)), y=0+1*sin(seq(0,2*pi,length.out=100)))
ggplot()+aes(x=G1,y=G2)+circle+
  geom_point()+
  coord_fixed(ratio=1)+ #pour ne pas afficher une ellipse
  geom_segment(aes(x=0,y=0,xend=G1,yend=G2),arrow=arrow(),size=1)+
  ggrepel::geom_label_repel(aes(label=var_var))


#Comparaison avec la commande PCA
res = PCA(raisin[,-p],scale.unit=TRUE)

#Question 3 : Classification hiérarchique ascendante
d.data <- dist(X)
hc <- hclust(d.data)
#dendogramme
plot(hc)
rect.hclust(hc, k = 2)
cut <- cutree(hc,k=2)
sil <- silhouette(cut,d.data)
mean(sil[,3]) #l'indice silhouette est seulement dans la troisième colonne
fviz_silhouette(sil)
plt1 <- fviz_cluster(object=list(data = X, cluster = cut))
#Indice de sihoulette proche de 0.5 et très peu de négatif, un clustering à deux groupes est satisfaisant

#Pour comparer avec la partition originelle, on peut utiliser un indice de Ward
library(mclust)
Class <- raisin$Class
adjustedRandIndex(cut,Class)

#Question 4
F3 <- X_mat%*%eigen(cor(X))$vectors[,3]
d.data <- dist(F1+F2+F3)
hc <- hclust(d.data)
cut <- cutree(hc,k=2)
adjustedRandIndex(cut,Class)
#centrage et reduction
raisin=cbind(X,raisin[,p])
#chantillons d"apprentissage
set.seed(1)
train = sample(c(TRUE,FALSE),n,rep=TRUE,prob = c(2/3,1/3))
train_data <- raisin[train, ]
test_data  <- raisin[!train, ]
appre = PCA(train_data[,-p])
train_pca_scores <- appre$ind$coord[, 1:2]
test_pca_scores <- predict(appre, newdata = test_data[, -p])$coord[, 1:2]
train_data_pca <- data.frame(PC1 = train_pca_scores[, 1], PC2 = train_pca_scores[, 2],Class = train_data$Class)
test_data_pca <- data.frame(PC1 = test_pca_scores[, 1], PC2 = test_pca_scores[, 2],Class = test_data$Class)
proj_test<- test_pca_scores[,1]
proj_test
#logistique
train_data$Class <- ifelse(train_data$Class == "Kecimen",
                           1,
                           ifelse(train_data$Class == "Besni", 0, NA))
train_data_pca$Class <- ifelse(train_data_pca$Class == "Kecimen",
                           1,
                           ifelse(train_data_pca$Class == "Besni", 0, NA))
test_data$Class <- ifelse(test_data$Class == "Kecimen",
                           1,
                           ifelse(test_data$Class == "Besni", 0, NA))
test_data_pca$Class <- ifelse(test_data_pca$Class == "Kecimen",
                          1,
                          ifelse(test_data_pca$Class == "Besni", 0, NA))
res.glm = glm(Class~.,
              family=binomial,data=train_data)
residuals(res.glm, type = "response")
res.glm2 <- glm(Class ~ PC1 + PC2, family = binomial(link = "logit"), data = train_data_pca)
library(MASS)
aicchoose= stepAIC(res.glm)
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$Class                            
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
# alpha = 1 lasso
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = cv_lasso$lambda.min)
cat("\nOptimal lambda for Lasso:", cv_lasso$lambda.min, "\n")
#SVM
library(e1071)
svm_linear <- svm(Class ~ ., data = train_data, kernel = "linear", probability = TRUE)
svm_poly <- svm(Class ~ ., data = train_data, kernel = "polynomial", degree = 3, probability = TRUE)
library(pROC)
pred_glm_train <- predict(res.glm, newdata = train_data[,-p], type = "response")
roc_glm_train <- roc(train_data$Class, pred_glm_train)
# predict values
pred_glm_test <- predict(res.glm, newdata = test_data, type = "response")
roc_glm_test <- roc(test_data$Class, pred_glm_test)
pred_glm2_test <- predict(res.glm2, newdata = test_data_pca, type = "response")
roc_glm2_test <- roc(test_data_pca$Class, pred_glm2_test)

pred_aic_test <- predict(aicchoose, newdata = test_data, type = "response")
roc_aic_test <- roc(test_data$Class, pred_aic_test)

x_test <- as.matrix(test_data[, -ncol(test_data)])
pred_lasso_test <-  as.vector(predict(lasso_model, newx = x_test, type = "response", s = cv_lasso$lambda.min))
roc_lasso_test <- roc(test_data$Class, as.vector(pred_lasso_test))

pred_svm_linear_test <-predict(svm_linear, newdata = test_data, probability = TRUE)
roc_svm_linear_test <- roc(test_data$Class, pred_svm_linear_test)

pred_svm_poly_test <- predict(svm_poly, newdata = test_data, probability = TRUE)
roc_svm_poly_test <- roc(test_data$Class, pred_svm_poly_test)

# ROC
par(mfrow = c(1, 1))
plot(roc_glm_train, col = "blue", lty = 1, main = "ROC Curves (Full GLM on Train, Others on Test)", 
     lwd = 2, print.auc = TRUE, print.auc.y = 0.5, print.auc.x = 0.2, print.auc.col = "blue")
plot(roc_glm_test, col = "blue", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.5, print.auc.x = 0.8, print.auc.col = "blue")
plot(roc_glm2_test, col = "red", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.45, print.auc.x = 0.8, print.auc.col = "red")
plot(roc_aic_test, col = "green", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.4, print.auc.x = 0.8, print.auc.col = "green")
plot(roc_lasso_test, col = "purple", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.35, print.auc.x = 0.8, print.auc.col = "purple")
plot(roc_svm_linear_test, col = "orange", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.3, print.auc.x = 0.8, print.auc.col = "orange")
plot(roc_svm_poly_test, col = "black", lty = 2, add = TRUE, lwd = 2, print.auc = TRUE, print.auc.y = 0.25, print.auc.x = 0.8, print.auc.col = "black")

legend("bottomright", 
       legend = c("Full GLM (Train)", "Full GLM (Test)", 
                  "PCA GLM (Test)", 
                  "AIC GLM (Test)", 
                  "Lasso GLM (Test)", 
                  "SVM Linear (Test)", 
                  "SVM Polynomial (Test)"),
       col = c("blue", "blue", "red", "green", "purple", "orange", "black"),
       lty = c(1, 2, 2, 2, 2, 2, 2),  # 实线表示训练集，虚线表示测试集
       lwd = 2)

# AUC
cat("\nAUC on Training Set (Full GLM only):\n")
cat("Full GLM:", auc(roc_glm_train), "\n")

cat("\nAUC on Test Set:\n")
cat("Full GLM:", auc(roc_glm_test), "\n")
cat("PCA GLM:", auc(roc_glm2_test), "\n")
cat("AIC GLM:", auc(roc_aic_test), "\n")
cat("Lasso GLM:", auc(roc_lasso_test), "\n")
cat("SVM Linear:", auc(roc_svm_linear_test), "\n")
cat("SVM Polynomial:", auc(roc_svm_poly_test), "\n")
###les erreurs
##glm modele(Règle de Bayes)
#glm complet
bayes = mean(train_data$Class == 0)
trainpred_glm<- ifelse(pred_glm_train-bayes>=0,
                               1,
                               ifelse(pred_glm_train-bayes<0, 0, NA))
sum(abs(train_data$Class-trainpred_glm))
bayes2 = mean(test_data$Class == 0)
testpred_glm<- ifelse(pred_glm_test-bayes2>=0,
                       1,
                       ifelse(pred_glm_test-bayes2<0, 0, NA))
sum(abs(test_data$Class-testpred_glm))
#glm partiel
trainpred_glm2<- ifelse(res.glm2[["fitted.values"]]-bayes>=0,
                       1,
                       ifelse(res.glm2[["fitted.values"]]-bayes<0, 0, NA))
sum(abs(train_data_pca$Class-trainpred_glm2))
testpred_glm2<- ifelse(pred_glm2_test-bayes2>=0,
                      1,
                      ifelse(pred_glm2_test-bayes2<0, 0, NA))
sum(abs(test_data_pca$Class-testpred_glm2))###why diminue?
#aic
trainpred_aic<- ifelse(aicchoose[["fitted.values"]]-bayes>=0,
                        1,
                        ifelse(aicchoose[["fitted.values"]]-bayes<0, 0, NA))
sum(abs(train_data$Class-trainpred_aic))
testpred_aic<- ifelse(pred_aic_test-bayes2>=0,
                       1,
                       ifelse(pred_aic_test-bayes2<0, 0, NA))
sum(abs(test_data$Class-testpred_aic))
##lasso
x_train <- as.matrix(train_data[, -ncol(train_data)])
pred_lasso_train <-  as.vector(predict(lasso_model, newx = x_train, type = "response", s = cv_lasso$lambda.min))
trainpred_lasso<- ifelse(pred_lasso_train-bayes>=0,
                       1,
                       ifelse(pred_lasso_train-bayes<0, 0, NA))
sum(abs(train_data$Class-trainpred_aic))
testpred_lasso<- ifelse(pred_lasso_test-bayes2>=0,
                      1,
                      ifelse(pred_lasso_test-bayes2<0, 0, NA))
sum(abs(test_data$Class-testpred_lasso))
##svm
#linear
pred_labels_test = ifelse(pred_svm_linear_test >= 0, 1, 0)
sum(abs(pred_labels_test- test_data$Class))
pred_labels_train = ifelse(svm_linear[["decision.values"]] >= 0, 1, 0)
sum(pred_labels_train != train_data$Class)#mauvais en test
#poly
pred_labels_test2 = ifelse(pred_svm_poly_test >= 0, 1, 0)
sum(pred_labels_test2 != test_data$Class)
pred_labels_train2 = ifelse(svm_poly[["decision.values"]] >= 0, 1, 0)
sum(pred_labels_train2 != train_data$Class)#mauvais en test
###Parti 3
n1 <- sum(train_data_pca$Class==1)
n2 <- sum(train_data_pca$Class==0)
S1 <- cov(train_data_pca[which(train_data_pca$Class==1), 1:2])
S2 <- cov(train_data_pca[which(train_data_pca$Class==0), 1:2])
sigema <- ((n1-1)*S1+(n2-1)*S2)/(n1+n2-2)
P0=mean(train_data_pca$Class==0)
P1=mean(train_data_pca$Class==1)
res.lda=lda(Class ~ PC1 + PC2,data = train_data_pca)
inverse=solve(sigema)
pi=function(x)
{return (P0)*exp(-1/2*t(x-as.matrix(res.lda$means[1,]))%*%inverse%*%(x-as.matrix(res.lda$means[1,])))+(P1)*exp(-1/2*t(x-as.matrix(res.lda$means[2,]))%*%inverse%*%(x-as.matrix(res.lda$means[2,])))}
pic0=function(x)
{return (P0)*exp(-1/2*t(x-as.matrix(res.lda$means[1,]))%*%inverse%*%(x-as.matrix(res.lda$means[1,])))/pi(x)}
pic1=function(x)
{return (P1)*exp(-1/2*t(x-as.matrix(res.lda$means[2,]))%*%inverse%*%(x-as.matrix(res.lda$means[2,])))/pi(x)}
score0=function(x)
{t(x)%*%inverse%*%as.matrix(res.lda$means[1,])-1/2*t(as.matrix(res.lda$means[1,]))%*%inverse%*%as.matrix(res.lda$means[1,])+log(pic0(x))}
score1=function(x)
{t(x)%*%inverse%*%as.matrix(res.lda$means[2,])-1/2*t(as.matrix(res.lda$means[2,]))%*%inverse%*%as.matrix(res.lda$means[2,])+log(pic1(x))}
x_range <- range(train_data_pca[,1])
y_range <- range(train_data_pca[,2])
x_seq <- seq(x_range[1], x_range[2], length.out = 100)
y_seq <- seq(y_range[1], y_range[2], length.out = 100)
grid <- expand.grid(x = x_seq, y = y_seq)
calcul <- t(as.matrix(grid))
datascore0 <- apply(calcul, 2, score0)
datascore1 <- apply(calcul, 2, score1)
score_diff <- datascore1 - datascore0
score_matrix <- matrix(score_diff, nrow = 100, byrow = FALSE)
plot(train_data_pca[,1:2], col = train_data_pca[,3] + 1, pch = 16)
contour(x_seq, y_seq, score_matrix, levels = 0, drawlabels = FALSE, add = TRUE, col = "blue")
###Σ=QΛQ 
Q=eigen(sigema)
score0(t(as.matrix((train_data_pca[1,-3]))))
inverse=Q$vectors%*%diag(1/Q$values)%*%t(Q$vectors)
score0(t(as.matrix((train_data_pca[1,-3]))))

pred_labels <- apply(t(as.matrix(test_data_pca[, 1:2])), 2, function(x) {
  if (score0(x) > score1(x)) {
    return(0)
  } else {
    return(1)
  }
})


true_labels <- test_data_pca[, 3]

error_rate <- mean(pred_labels != true_labels)
print(paste("Erreur de classification (méthode manuelle):", round(error_rate, 4)))
pred_lda_test <- predict(res.lda, newdata = test_data_pca)
error_rate_lda <- mean(pred_lda_test$class != test_data_pca$Class)
print(paste("Erreur de classification (avec lda()):", round(error_rate_lda, 4)))
###ROC
library(pROC)
roc_plot=roc(true_labels, pred_lda_test$posterior[, 2])
plot(roc_plot, col = "blue", main = "Courbe ROC - LDA,QDA")
auc(roc_plot)
##quadratique
res.qda = qda(Class ~ PC1 + PC2,data = train_data_pca)
pred_qda_test=predict(res.qda, newdata =test_data_pca)
1-mean(test_data_pca$Class==pred_qda_test$class)
roc_plot2=roc(true_labels, pred_qda_test$posterior[, 2])
par(mfrow = c(1, 1))
plot(roc_plot, col = "blue", lty =1, add = FALSE, lwd = 2, print.auc = TRUE,print.auc.y = 0.5, print.auc.x = 0.8, print.auc.col = "blue",main = "Courbe ROC - LDA,QDA")
plot(roc_plot2, col = "green",lty = 2, add = TRUE, lwd = 2, print.auc = TRUE,print.auc.y = 0.45, print.auc.x = 0.8, print.auc.col = "green")
auc(roc_plot2)
##complete lda
res.lda2=lda(Class ~.,data = train_data)
pred_lda2_test=predict(res.lda2, newdata =test_data)
1-mean(test_data$Class==pred_lda2_test$class)
roc_plot3=roc(true_labels, pred_lda2_test$posterior[, 2])
plot(roc_plot3, col = "red",lty = 2, add = TRUE, lwd = 2, print.auc = TRUE,print.auc.y = 0.4, print.auc.x = 0.8, print.auc.col = "red")
auc(roc_plot3)
