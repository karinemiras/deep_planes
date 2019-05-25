 
library(ggplot2)
library(sqldf)

path = '/Users/kdo210/Documents/deeplearning/plane_deep/'
methods = c('model_plane_0409',
            'model_plane_040905',
            'model_plane_080905',
            'model_plane_04071',
            'model_plane_040705',
            'model_plane_04091',
            'model_plane_03091',
            'model_plane_060705',
             'model_plane_05071')

opt_methods = c()

for (m in 1:length(methods))
{
 
    opt   = read.table(paste(path,'models/', methods[m],".txt", sep=''), header = FALSE)
    opt$Model = methods[m] 
    opt_methods = rbind(opt_methods, opt)

}



train = sqldf("select Model, V1, V2,  avg(V4) as CE from opt_methods
                         where V1 = 'train' group by 1,2,3 ")



graph <- ggplot(train, aes(V2, CE)) + 
         geom_line(aes(color = Model), size=0.7) + 
         labs( title="Training", x="Step" )+ 
         coord_cartesian(ylim = c(0, 10))+
         theme( plot.title=element_text(size=30 ), legend.text=element_text(size=15),
                axis.text=element_text(size=20),axis.title=element_text(size=25) ) 

ggsave(paste( path ,'analysis/opt_train.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)


 