 
library(ggplot2)
library(sqldf)

path = '/Users/kdo210/Documents/deeplearning/plane_deep/'

methods = c('old/over')

methods = c('tuning/model_plane_0409',
            'tuning/model_plane_040905',
            'tuning/model_plane_080905',
            'tuning/model_plane_04071',
            'tuning/model_plane_040705',
            'tuning/model_plane_04091',
            'tuning/model_plane_03091',
            'tuning/model_plane_060705',
             'tuning/model_plane_05071'
            )

methods = c(
            'candidate_step3/model_plane_03091',
            'finetune1_step2/model_plane_030915',
            'finetune2_step2/model_plane_0309125'
            #,'finetune4_step1/model_plane_030905'
            #, 'finetune3_step1/model_plane_005091'
)


methods = c(
  'candidate_step3/model_plane_03091'
)

opt_methods = c()

for (m in 1:length(methods))
{
 
    opt   = read.table(paste(path,'models/', methods[m],".txt", sep=''), header = FALSE)
    #opt = sqldf("select * from opt where V1='train'")
    
   
      counter = 0
      epoch = 0
      for (i in 1:nrow(opt))
      {
        opt[i,]$V2 = epoch
        counter = counter +1
       # if(counter == 1){
        if(counter == 47){
          epoch = epoch +1
          counter = 0
        }
      }
    
    
    opt$Model = methods[m] 
    opt_methods = rbind(opt_methods, opt)

}



train = sqldf("select Model, V1, V2,  avg(V4) as CE from opt_methods
                         where V1 = 'train' and V2<= 132 group by 1,2,3 ")


graph <- ggplot(train, aes(V2, CE)) + 
         geom_line(aes(color = Model), size=2, alpha=0.7) + 
         labs( title="Training", x="Step" )+ 
         coord_cartesian(ylim = c(0, 10))+
         theme( plot.title=element_text(size=30 ), legend.text=element_text(size=15),
                axis.text=element_text(size=20),axis.title=element_text(size=25) ) 

#ggsave(paste( path ,'analysis/over.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)
#ggsave(paste( path ,'analysis/ce_tuning.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)
ggsave(paste( path ,'analysis/ce_finetuning.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)

########
# fine tuning

samples = c(
  'planetrain_model_plane_04_09_0',
  'planetrain_model_plane_04_09_05',
  'planetrain_model_plane_08_09_05',
  'planetrain_model_plane_04_07_1',
  'planetrain_model_plane_04_07_05',
  'planetrain_model_plane_04_09_1',
  'planetrain_model_plane_03_09_1',
  'planetrain_model_plane_06_07_05',
  'planetrain_model_plane_05_07_1'
)


samples = c(
            'planeval_model_plane_04_09_0',
            'planeval_model_plane_04_09_05',
            'planeval_model_plane_08_09_05',
            'planeval_model_plane_04_07_1',
            'planeval_model_plane_04_07_05',
            'planeval_model_plane_04_09_1',
            'planeval_model_plane_03_09_1',
            'planeval_model_plane_06_07_05',
            'planeval_model_plane_05_07_1'
            )

samples = c(
  'planeval_model_plane_0309125',
  'planeval_model_plane_005091'
)


measures = c()

for (s in 1:length(samples))
{
  measure   = read.table(paste(path,'analysis/tuning/', samples[s],"_measures.txt", sep=''), header = FALSE)
  measure$lr = strsplit(samples[s],'_')[[1]][4]
  measure$lr_ = 0
  measure$lrd = strsplit(samples[s],'_')[[1]][5]
  measure$lrd_ = 0
  measure$l2 = strsplit(samples[s],'_')[[1]][6]
  measure$l2_ = 0
  measure$model = strsplit(samples[s],'_plane_')[[1]][2]
  measures = rbind(measures, measure)
}



graph = ggplot(data=measures, aes(x=V1, y=V2,  fill=model)) +
  geom_bar(stat="identity", position = "dodge")+
  #labs( title="Validation", x="Measure", y="Value" )+ 
  labs( title="Training", x="Measure", y="Value" )+ 
  theme( plot.title=element_text(size=30 ), legend.text=element_text(size=15),
         axis.text=element_text(size=20),axis.title=element_text(size=25) ) 
ggsave(paste( path ,'analysis/measures_tuning.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)
ggsave(paste( path ,'analysis/measures_train_tuning.pdf',  sep=''), graph , device='pdf', height = 8, width = 10)


fscores = sqldf("select * from measures where V1 = 'fscore'")


for (i in 1:nrow(fscores))
{
  if(fscores[i,]$l2 == '05'){
    fscores[i,]$l2_ = 0.0005
  }else if(fscores[i,]$l2 == '1'){
    fscores[i,]$l2_ = 0.001
  }else{
    fscores[i,]$l2_ = 0
  }
  
  if(fscores[i,]$lrd == '07'){
    fscores[i,]$lrd_ = as.numeric(0.7)
  }else if(fscores[i,]$lrd == '09'){
    fscores[i,]$lrd_ = as.numeric(0.9)
  } 
  
  if(fscores[i,]$lr == '03'){
    fscores[i,]$lr_ = as.numeric(0.001)
  }
  if(fscores[i,]$lr == '04'){
    fscores[i,]$lr_ = as.numeric(0.0001)
  }
  if(fscores[i,]$lr == '05'){
    fscores[i,]$lr_ = as.numeric(0.00001)
  }
  if(fscores[i,]$lr == '06'){
    fscores[i,]$lr_ = as.numeric(0.000001)
  }
  if(fscores[i,]$lr == '08'){
    fscores[i,]$lr_ = as.numeric(0.00000001)
  }
}

r = lm(formula = V2 ~  
         (lr_ + lrd_ + l2_
         ) ^3
       ,data=fscores )
print(summary(r))





