# 01/12/2022
# process simulation results

# 01/24/2022
# plot posterior medians instead

# 01/25/2022
# get posterior CIs as well

# 07/2023
# include new simulations using the submodel (fixed thresholds on point types)

library(tidyverse)
library(ggplot2)
library(ggpubr)
#setwd("~/Documents/Research_and_References/HIV_transmission_flow/")
setwd('~/Documents/Research/HIV_transmission_flow/')

# flag: use sub-model results??
submodel = FALSE
if(submodel){
  suffix = "_submodel"
}else{
  suffix = ""
}

## Posterior means
# previous simulation with N=400
# dat_weight1 = read_csv('weight_means.csv')
# dat_prop1 = read_csv('prop_means.csv')

# new simulations with N=100,200,600,800
# dat_weight2 = read_csv('pooled_weights.csv')
# dat_prop2 = read_csv('pooled_Cs.csv')

## Posterior medians with N = 400 runs
dat_weight1 = read_csv(sprintf('pooled_median_weights_400%s.csv', suffix))
dat_prop1 = read_csv(sprintf('pooled_median_Cs_400%s.csv', suffix))

# new simulations with N=100,200,600,800
dat_weight2 = read_csv(sprintf('pooled_median_weights%s.csv', suffix))
dat_prop2 = read_csv(sprintf('pooled_median_Cs%s.csv', suffix))

## results with posterior CIs
# previous simulations with N=400
# dat_weight1 = read_csv('pooled_weights_CIs_400.csv')
# dat_prop1 = read_csv('pooled_Cs_CIs_400.csv')

# new simulations with N=100,200,600,800
# dat_weight2 = read_csv('pooled_weights_CIs.csv')
# dat_prop2 = read_csv('pooled_Cs_CIs.csv')


# ONLY used for the initial results with means
# # put them together
# dat_weight1 = dat_weight1 %>% mutate(N = 400) %>%
#   select(N, scenario, weight, means)
# dat_prop1 = dat_prop1 %>% mutate(N = 400) %>%
#   select(N, scenario, weight, means)

# remove those 0 entries (filled in for missing values...)
dat_weight = rbind(dat_weight1, dat_weight2) %>% 
  filter(means > 0) %>%
  arrange(N)

dat_prop = rbind(dat_prop1, dat_prop2) %>% 
  filter(means > 0) %>% arrange(N)

# save results
saveRDS(dat_weight, sprintf("male_source_weight_sim_summary%s.rds", suffix))
saveRDS(dat_prop, sprintf("surf_prop_sim_summary%s.rds", suffix))


## 01/25/2022
## ONLY for summary table with credible intervals ##
# dat_weight = rbind(dat_weight1, dat_weight2) %>% 
#   filter(means > 0) %>%
#   arrange(N)
# 
# dat_prop = rbind(dat_prop1, dat_prop2) %>% 
#   filter(means > 0) %>% arrange(N)

## create data table with ground truth
# weight_truth = data.frame(scenario = c(1,1,2,2),
#                           weight = c('younger', 'older', 'younger', 'older'),
#                           truth = c(0.3, 0.6, 0.6, 0.3))
# prop_truth = data.frame(scenario = c(1,1,2,2),
#                         weight = c('M->F', 'F->M', 'M->F', 'F->M'),
#                         truth = c(0.5, 0.5, 0.6, 0.4))

## calculate coverage
# weight_coverage = dat_weight %>% 
#   inner_join(weight_truth) %>%
#   mutate(cover = (ubs >= truth) & (lbs <= truth)) %>%
#   group_by(N, scenario, weight) %>%
#   filter(means >= quantile(means, 0.08), 
#          means <= quantile(means, 0.92)) %>%
#   summarize(coverage_rate = mean(cover))
# 
# prop_coverage = dat_prop %>% 
#   inner_join(prop_truth) %>%
#   mutate(cover = (ubs >= truth) & (lbs <= truth)) %>%
#   group_by(N, scenario, weight) %>%
#   filter(means >= quantile(means, 0.05), 
#          means <= quantile(means, 0.95)) %>%
#   summarize(coverage_rate = mean(cover))

## save CI coverage results
# saveRDS(weight_coverage, 'male_source_weight_post_coverage.rds')
# saveRDS(prop_coverage,'surf_prop_post_coverage.rds')

# try plotting

## read in saved RDS summary tables
suffix = "_submodel"
#suffix = ""
dat_weight = readRDS(sprintf("male_source_weight_sim_summary%s.rds", suffix))
dat_prop = readRDS(sprintf("surf_prop_sim_summary%s.rds", suffix))

# pdf file to save plots
pdf('sim_plots_diff_Ns.pdf', height = 4, width = 6)

## 1. props of younger vs older male sources
ylims = c(0,1)
### v1: more older source
ggplot(dat_weight %>% filter(scenario==1)) +
  geom_hline(yintercept = c(0.62, 0.32), color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_y_continuous(limits = ylims)+
  labs(x='event size N', y='post. mean proportions', 
       fill='male \ntransmitter', title='scenario 1: more older transmitters') +
  theme_bw(base_size = 14)

### v2: more younger source
ggplot(dat_weight %>% filter(scenario==2)) +
  geom_hline(yintercept = c(0.32, 0.62), color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_y_continuous(limits = ylims)+
  labs(x='event size N', y='post. mean proportions', 
       fill='male \ntransmitter', title='scenario 2: more younger transmitters') +
  theme_bw(base_size = 14)

## 2. props of MF vs FM transmissions
ylims = c(0.3, 0.7)
### v1: 50-50 
ggplot(dat_prop %>% filter(scenario==1)) +
  geom_hline(yintercept = c(0.5), color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_y_continuous(limits = ylims)+
  scale_fill_manual(values = c("#7CAE00", "#C77CFF"))+
  labs(x='event size N', y='post. mean proportions', 
       fill='transmission\ndirection', 
       title='scenario 1: equal proportions') +
  theme_bw(base_size = 14)#+
  #theme(legend.position = 'none')

### v2: 60-40 MF-FM
ggplot(dat_prop %>% filter(scenario==2)) +
  geom_hline(yintercept = c(0.6, 0.4), color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_fill_manual(values = c("#7CAE00", "#C77CFF"))+
  scale_y_continuous(limits = ylims)+
  
  labs(x='event size N', y='post. mean proportions', 
       fill='transmission\ndirection', 
       title='scenario 2: 60-40 split, more M->F') +
  theme_bw(base_size = 14)

dev.off()



#####
# use facets for plotting instead

# 01/24/2022: change y-axis label to median

# pdf('joint_sim_plots_diff_Ns.pdf', 
#     height = 4, width = 10)

# pdf('joint_sim_plots_diff_Ns_median.pdf', 
#     height = 4, width = 10)

# 04/24/2022: figure format updates (per Oliver's suggestion)
pdf('joint_sim_plots_diff_Ns_median_update.pdf', 
    height = 6.7, width = 10)

# 07/19/2022: put MF vs FM proportions on top row
pdf('joint_sim_plots_diff_Ns_median_update2.pdf', 
    height = 6.7, width = 10)

# 12/08/2022: re-make plots with new facet titles and annotation terms...
pdf('joint_sim_plots_diff_Ns_median_update3.pdf', 
    height = 6.7, width = 10)


## 1. younger vs. older weights
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.62, 0.33, 0.33, 0.62))
# sce.labs = c('Scenario 1: same age transmission',
#              'Scenario 2: discordant age transmission')
sce.labs = c('Simulation scenario: SAME AGE',
             'Simulation scenario: DISCORDANT AGE')
names(sce.labs) = c('1','2')

ylims = c(0, 1)

pweight = 
ggplot(dat_weight) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  scale_fill_discrete(labels = c("age difference \n+/- 5 years",
                                 "age difference \n>5 years"))+
  labs(x='sample size N\n(number of likely transmission pairs)', 
       y='estimated source proportions\n(posterior medians in 100 replicates)', 
       fill='sources of\ninfections\nin adolescent\nand young \nwomen\n') +
  theme_bw(base_size = 14) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))



## 2. MF vs. FM proportions 
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.5, 0.5, 0.6, 0.4))
# sce.labs = c('Scenario 1: MF 50-50',
#              'Scenario 2: MF 60-40')
sce.labs = c('Simulation scenario: MF 50-50',
             'Simulation scenario: MF 60-40')
names(sce.labs) = c('1','2')

ylims = c(0.3, 0.7)
#ylims = c(0,1)

pprop = 
ggplot(dat_prop) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = 0.5) +
  scale_fill_manual(values = c("#7CAE00", "#C77CFF"))+
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  labs(x='sample size N\n(number of likely transmission pairs)', 
       y='estimated source proportions\n(posterior medians in 100 replicates)', 
       fill='transmission \ndirection') +
  theme_bw(base_size = 14) + 
  facet_grid(~scenario, 
             labeller = labeller(
               scenario = sce.labs))


# combined_figure = 
# ggpubr::ggarrange(pweight+rremove("xlab")+rremove("x.text")+
#                     rremove('x.ticks') + rremove('ylab'),
#                   pprop+rremove('ylab'),
#                   nrow = 2,
#                   heights = c(5,6),
#                   align = 'v')
combined_figure = 
  ggpubr::ggarrange(pprop+rremove("xlab")+rremove("x.text")+
                      rremove('x.ticks') + rremove('ylab'),
                    pweight+rremove('ylab'),
                    nrow = 2,
                    heights = c(5,6),
                    align = 'v')
annotate_figure(combined_figure, 
                left = grid::textGrob('estimated source proportions\n(posterior medians in 100 replicates)', 
                                      rot = 90, vjust = 0.5, 
                                      gp = grid::gpar(cex = 1.2)))

dev.off()


#####
# July 2023: plots with full-model and sub-model together
suffix = "_submodel"
dat_weight_sub = readRDS(sprintf("male_source_weight_sim_summary%s.rds", suffix)) %>%
  mutate(weight = if_else(weight == "younger", "younger sub", "older sub"))
dat_prop_sub = readRDS(sprintf("surf_prop_sim_summary%s.rds", suffix)) %>%
  mutate(weight = if_else(weight == "M->F", "M->F sub", "F->M sub"))

suffix = ""
dat_weight_full = readRDS(sprintf("male_source_weight_sim_summary%s.rds", suffix)) %>%
  mutate(weight = if_else(weight == "younger", "younger full", "older full"))
dat_prop_full = readRDS(sprintf("surf_prop_sim_summary%s.rds", suffix)) %>%
  mutate(weight = if_else(weight == "M->F", "M->F full", "F->M full"))

dat_weight = bind_rows(dat_weight_full, dat_weight_sub)
dat_prop = bind_rows(dat_prop_full, dat_prop_sub)


# re-produce plot, comparing full and subset models
pdf('joint_sim_plots_diff_Ns_median_compare_models.pdf', 
    height = 6.7, width = 10.5)

boxWidth = 0.7

## 1. younger vs. older weights
dummy = data.frame(scenario = rep(1:2, each = 2), 
                   yinter = c(0.62, 0.33, 0.33, 0.62))
# sce.labs = c('Scenario 1: same age transmission',
#              'Scenario 2: discordant age transmission')
sce.labs = c('Simulation scenario: SAME AGE',
             'Simulation scenario: DISCORDANT AGE')
names(sce.labs) = c('1','2')

boxLabels = c("+/- 5 years (FULL)",
              "+/- 5 years (SUBSET)",
              ">5 years (FULL)",
              ">5 years (SUBSET)")
cols = c(wes_palette("Darjeeling2")[2], # dark blue
         wes_palette("Moonrise3")[1],  # light blue
         wes_palette("GrandBudapest1")[2],  # dark pink
         wes_palette("Moonrise3")[2]) # light pink


ylims = c(0, 1)

pweight = 
  ggplot(dat_weight) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = boxWidth) +
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  scale_fill_manual(labels = boxLabels,
                    values = cols)+
  geom_vline(xintercept = c(1.5, 2.5, 3.5, 4.5), 
             color = "gray70", linetype = 1) +
  labs(x='sample size N\n(number of potential transmission pairs)', 
       y='estimated source proportions\n(posterior medians in 100 replicates)', 
       fill='sources of\ninfections in \nadolescent and \nyoung women:\nage difference') +
  theme_bw(base_size = 15) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))

## 2. MF vs. FM proportions 
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.5, 0.5, 0.6, 0.4))
# sce.labs = c('Scenario 1: MF 50-50',
#              'Scenario 2: MF 60-40')
sce.labs = c('Simulation scenario: MF 50-50',
             'Simulation scenario: MF 60-40')
names(sce.labs) = c('1','2')

ylims = c(0.3, 0.7)
#ylims = c(0,1)

boxLabels = c("F->M (FULL)",
              "F->M (SUBSET)",
              "M->F (FULL)",
              "M->F (SUBSET)")

cols = c(wes_palette("IsleofDogs1")[1], # violet?
         wes_palette("GrandBudapest2")[2], # light violet?
         wes_palette("Chevalier1")[1], # dark green
         wes_palette("Moonrise3")[3]) # light green?

cols = c(wes_palette("Darjeeling1")[2],
         wes_palette("Moonrise2")[3],
         wes_palette("Darjeeling1")[4],
         wes_palette("Chevalier1")[2])

pprop = 
  ggplot(dat_prop) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray30', linetype =2)+
  geom_boxplot(aes(x=as.character(N), fill=weight, y=means), 
               outlier.shape = NA, width = boxWidth) +
  scale_fill_manual(labels = boxLabels,
                    values = cols)+
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  geom_vline(xintercept = c(1.5, 2.5, 3.5, 4.5), 
             color = "gray70", linetype = 1) +
  labs(x='sample size N\n(number of potential transmission pairs)', 
       y='estimated source proportions\n(posterior medians in 100 replicates)', 
       fill='transmission \ndirection') +
  theme_bw(base_size = 15) + 
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(~scenario, 
             labeller = labeller(
               scenario = sce.labs))

## combine together into one big plot
combined_figure = 
  ggpubr::ggarrange(pprop+rremove("xlab")+rremove("x.text")+
                      rremove('x.ticks') + rremove('ylab'),
                    pweight+rremove('ylab'),
                    nrow = 2,
                    heights = c(5,6),
                    align = 'v',
                    labels = c("A", "B"))
annotate_figure(combined_figure, 
                left = grid::textGrob('estimated source proportions\n(posterior medians in 100 replicates)', 
                                      rot = 90, vjust = 0.5, 
                                      gp = grid::gpar(cex = 1.2)))

dev.off()



##### 
# 01/25/2022: plot posterior credible intervals as well
# (using randomly chosen)

## 02/15/2022: update by including 5 examples for each scenario
pdf('joint_sim_plots_diff_Ns_CIs_5.pdf', 
    height = 4, width = 10)

## 04/24/2022: update plot labels of the CIs
pdf('joint_sim_plots_diff_Ns_CIs_5_update.pdf', 
    height = 7, width = 10)

set.seed(41) # or some other random seeds

## 1. younger vs. older weights
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.6, 0.3, 0.3, 0.6))
sce.labs = c('Scenario 1: same age transmissions',
             'Scenario 2: discordant age transmissions')
names(sce.labs) = c('1','2')

ylims = c(0, 1)

### randomly select one result for each N
### Feb 15 2022: select 3 or 5 cases to showcase
to_sel = 5

rand_weights <- dat_weight %>%  
  inner_join(weight_truth) %>%
  filter((ubs >= truth) & (lbs <= truth)) %>%
  group_by(N, scenario, weight) %>%
  sample_n(to_sel) %>%
  mutate(group = 1:to_sel) %>% # add grouping label
  ungroup()

CIs_weights = 
ggplot(rand_weights, aes(x = N, color = weight, group=group)) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray50', linetype =2)+
  geom_errorbar(aes(ymax = ubs, ymin = lbs), 
                width = 50, size = 0.8,
                position =  position_dodge(width = 50)) +
  # geom_point(aes(y = means),
  #            position =  position_dodge(width = 25)) +
  scale_x_continuous(breaks = c(100,200,400,600,800)) +
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  scale_color_discrete(labels = c("age difference \n+/- 5 years",
                                  "age difference \n>5 years"))+
  labs(x='placeholder', 
       y='', 
       color='sources of\ninfection in\nyoung women\n') +
  theme_bw(base_size = 14) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))


## 2. MF v.s. FM transmissions
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.5, 0.5, 0.6, 0.4))
sce.labs = c('Scenario 1: MF 50-50',
             'Scenario 2: MF 60-40')
names(sce.labs) = c('1','2')

ylims = c(0.3, 0.7)

## also randomly select one result for each N
## Feb 15: show more CIs for each scenario
to_sel = 5
rand_prop <- dat_prop %>%  
  inner_join(prop_truth) %>%
  filter((ubs >= truth) & (lbs <= truth)) %>%
  group_by(N, scenario, weight) %>%
  sample_n(to_sel) %>%
  mutate(group = 1:to_sel) %>%
  ungroup()

CIs_props = 
ggplot(rand_prop, aes(x=N, color=weight, group = group)) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray50', linetype =2)+
  geom_errorbar(aes(ymax = ubs, ymin = lbs), 
                width = 60, size = 0.8,
                position = position_dodge(width = 50)) +
  # geom_point(aes(y = means), 
  #            position =  position_dodge(width = 25)) +
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  scale_x_continuous(breaks = c(100,200,400,600,800)) +
  scale_color_manual(values = c("#7CAE00", "#C77CFF"))+
  labs(x='sample size N\n(number of likely transmission pairs)', 
       y='estimated source proportions & CIs', 
       color='transmission\ndirection') +
  theme_bw(base_size = 14) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))


combined_CIs = 
  ggpubr::ggarrange(CIs_weights+rremove("x.text")+
                      rremove('x.ticks')+rremove('xlab'),
                    CIs_props+rremove('ylab'),
                    nrow = 2,
                    heights = c(5,6),
                    align = 'v')
annotate_figure(combined_CIs, 
                left = grid::textGrob('95% CIs of estimated source proportions', 
                                      rot = 90, vjust = 1, 
                                      gp = grid::gpar(cex = 1.2)))


dev.off()


#### Feb 15 update ------------
#### Plot an example of one view and one scenario ------
## do this for the weight example
# 04/24/2022: update with labels
dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.6, 0.3, 0.3, 0.6))
sce.labs = c('Scenario 2: discordant age transmissions')
names(sce.labs) = c('2')

ylims = c(0, 1)

### randomly select one result for each N
### Feb 15 2022: select 3 or 5 cases to showcase
to_sel = 5

rand_weights2 <- dat_weight %>%  
  inner_join(weight_truth) %>%
  filter((ubs >= truth) & (lbs <= truth)) %>%
  group_by(N, scenario, weight) %>%
  sample_n(to_sel) %>%
  mutate(group = 1:to_sel) %>% # add grouping label
  ungroup() %>%
  filter(scenario==2)

ggplot(rand_weights2, aes(x = N, 
                          color = weight, group=group)) +
  geom_hline(yintercept = c(0.3,0.6), linetype = 2)+
  geom_errorbar(aes(ymax = ubs, ymin = lbs), 
                width = 50, size = 0.8,
                position =  position_dodge(width = 50)) +
  # geom_point(aes(y = means),
  #            position =  position_dodge(width = 25)) +
  scale_x_continuous(breaks = c(100,200,400,600,800)) +
  scale_y_continuous(limits = ylims,
                     labels = scales::percent)+
  scale_color_discrete(labels = c("age difference \n+/- 5 years",
                                 "age difference \n>5 years"))+
  labs(x='sample size N\n(number of likely transmission pairs)', 
       y='estimated source proportions & CIs', 
       color='sources of\ninfection in\nyoung women\n') +
  theme_bw(base_size = 14) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))

  

#####
# 01/27/2022: also plot coverage probabilities

## try a joint plot of weights & props
dat_coverage = bind_rows(weight_coverage, prop_coverage)

pdf('sim_CI_coverage_by_N.pdf', 
    height = 4, width = 10)

dummy = data.frame(scenario = rep(1:2, each = 2), yinter = c(0.5, 0.9, 0.5, 0.9))
sce.labs = c('Scenario 1: more older transmitters\n     equal MF/FM proportions',
             'Scenario 2: more younger transmitters\n     60-40 MF/FM proportions')
names(sce.labs) = c('1','2')

ylims = c(0, 1)

ggplot(dat_coverage, aes(x = N, y = coverage_rate, 
                         color = weight,
                         shape = weight)) +
  geom_hline(data = dummy, aes(yintercept = yinter), 
             color = 'gray50', linetype =2)+
  geom_line(position =  position_dodge(width = 50),
            size = 0.8) +
  geom_point(position =  position_dodge(width = 50), 
             size = 2.5) +
  scale_y_continuous(limits = ylims, 
                     breaks = c(0,0.25,0.5,0.75,0.9,1))+
  scale_x_continuous(breaks = c(100,200,400,600,800)) +
  scale_color_manual(limits = c('older','younger','F->M','M->F'),
                     labels = c('older transmitters',
                                'younger transmitters',
                                'F->M transmissions',
                                'M->F transmissions'),
                     values = c("#F8766D", "#00BFC4",
                                "#7CAE00", "#C77CFF")) +
  scale_shape_manual(limits = c('older','younger','F->M','M->F'),
                     labels = c('older transmitters',
                                'younger transmitters',
                                'F->M transmissions',
                                'M->F transmissions'),
                     values = c(15:18)) +
  labs(x='event size N', y='95% CI coverage rate', 
       color='proportions of ...',
       shape = 'proportions of ...') +
  theme_bw(base_size = 14) +
  facet_grid(~scenario, 
             labeller = labeller(scenario = sce.labs))

dev.off()
