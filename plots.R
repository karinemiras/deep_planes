 
library(ggplot2)
library(sqldf)

path = '/Users/kdo210/Documents/deeplearning/plane_deep/'
methods = c(#'model_plane_0409',
            #'model_plane_0609',
            #'model_plane_0407',
            #'model_plane_0607',
            #'model_plane_0507',
            #'model_plane_0509',
            #'test1',
            'over')

opt_methods = c()

for (m in 1:length(methods))
{
 
    opt   = read.table(paste(path,'models/', methods[m],".txt", sep=''), header = FALSE)
    opt$Model = methods[m] 
    opt_methods = rbind(opt_methods, opt)

}


#train = sqldf("select Model, V1, V2,  V3  as CE from opt_methods
 #                        where V1 = 'train'  ")

train = sqldf("select Model, V1, V2,  avg(V4) as CE from opt_methods
                         where V1 = 'train' group by 1,2,3 ")



graph <- ggplot(train, aes(V2, CE)) + 
         geom_line(aes(color = Model), size=0.7) + 
         labs( title="Training", x="Step" )+ 
         coord_cartesian(ylim = c(0, 10))+
         theme( plot.title=element_text(size=30 ), legend.text=element_text(size=15),
                axis.text=element_text(size=20),axis.title=element_text(size=25) ) 

ggsave(paste( path ,'analysis/opt_train.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)


#val = sqldf("select Model, V1, V2, avg(V3) as CE from opt_methods
 #                        where V1 = 'val' group by 1,2,3 ")

val = sqldf("select Model, V1, V2, avg(V4) as CE from opt_methods
                         where V1 = 'val'
                         group by 1,2,3 ")

graph <- ggplot(val, aes(V2, CE)) + 
  geom_line(aes(color = Model), size=1.5) + 
  labs( title="Validation",  x="Epochs" )+ 
  coord_cartesian(ylim = c(0, 10))+
  theme(  plot.title=element_text(size=30 ),
          legend.text=element_text(size=15), axis.text=element_text(size=20),axis.title=element_text(size=25) ) 

ggsave(paste( path ,'analysis/opt_val.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)


 