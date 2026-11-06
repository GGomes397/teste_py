library(readxl)
library(dplyr)

bruto <-read_excel("C:/Users/Acer/Documents/git/teste_py/mir fundo.xlsx")
bruto <- bruto[-1,]
predito <- bruto[,1]
glim
par(mfrow=c(6,5))
for (i in 2:26) {
  plot(bruto[,i], bruto$...1, 
       ylab = "Resposta")
  
}
glimpse(bruto)
transpoto<- t(bruto)

