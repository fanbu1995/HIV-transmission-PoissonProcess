# real data related plots
# cleaned up in July, 2022 for publication

# we have:
# 1. proportions of each type
# 2. spatial points colored with posterior type probs
# 3. source age distribution (marginal on male or female)
# 4. density of MF and FM surface, with contours for HDIs
# 5. data points on 2D age space, colored by posterior probability of type lables
# 6. combine the colored points (with post.probs) with contour lines of HDIs


library(dplyr)
library(bayestestR)
library(xtable)
library(wesanderson)

## should set wd it to the path of results files and real data files! 
# setwd('~/Documents/Research/HIV_transmission_flow/')


# 1. proportions of each type --------
probs = read_csv('real_data_probs_chains.csv')
fixed_probs = read_csv('fixed_real_data_probs_chains.csv')

burn = 1000

probs_summary = apply(probs[(burn+1):nrow(probs),], 2, 
                      function(v) c(mean(v, na.rm = TRUE),
                                    quantile(v, c(0.025, 0.975), na.rm = TRUE),
                                    sd(v, na.rm = TRUE)))

probs_summary = as.data.frame(t(probs_summary))
names(probs_summary) = c('avg', 'CI95_lb', 'CI95_ub', 'std')
probs_summary$prob = names(probs)

fixed_summary = data.frame(prob = names(fixed_probs),
                           avg = fixed_probs[1,] %>% as.numeric())

## 06/29/2022: output a summary table as well
comb_summary = cbind(probs_summary %>% select(prob, avg, CI95_lb, CI95_ub),
                     fixed_summary %>% select(fixed_avg = avg)) %>%
  mutate(model_N = round(avg * 526, digits = 0), 
         fixed_N = fixed_avg * 526) %>%
  mutate(model_avg_text = 
           sprintf('%.1f%% (%.1f%%, %.1f%%)', 
                   avg * 100, CI95_lb * 100, CI95_ub * 100),
         fixed_avg_text = sprintf('%.1f%%', fixed_avg * 100)) %>%
  select(prob, model_avg_text, model_N,
         fixed_avg_text, fixed_N)

print(xtable(comb_summary[c(2,3,1),]), 
      include.rownames = FALSE)


text_info = data.frame(x=names(probs),
                       y=0.8,
                       label = sprintf('fixed N=%.0f\n model N=%.0f',
                                       fixed_summary$avg * 526,
                                       probs_summary$avg * 526))


ggplot(probs_summary, aes(x=prob, y=avg)) +
  geom_bar(stat='identity', aes(color = prob, fill = prob))+
  geom_errorbar(aes(ymax = CI95_ub, ymin = CI95_lb), color='grey30', width = 0.5) +
  geom_point(data=fixed_summary, shape = 17, aes(x=prob, y=avg), size = 4)+
  geom_text(data=text_info, aes(x=x, y=y, label=label), hjust=0.5, size = 5)+
  scale_y_continuous(limits = c(0,1), labels = scales::percent) +
  scale_x_discrete(labels = c('none', 'M->F', 'F->M'))+
  scale_fill_manual(values = c('grey70',wes_palette("Moonrise3")[1:2]))+
  scale_color_manual(values = c('grey70',wes_palette("Moonrise3")[1:2]))+
  labs(x='type', y='posterior mean & 95% CI')+
  theme_bw(base_size = 14) +
  theme(legend.position = 'none')





# 2. spatial points colored with posterior type probs-------
# omitted; the code is subsumed into part 5 and 6



# 3. source age distribution (marginal on male or female) ------
# 05/03 update: add age distribution from the fixed threshold analysis as well
males = read_csv('real_data_male_source_age_samples.csv')
females = read_csv('real_data_female_source_age_samples.csv')

malesFixed = read_csv('fixedThres_data_male_source_age_samples.csv')
femalesFixed = read_csv('fixedThres_data_female_source_age_samples.csv')

burn = 1000
thin = 20
maxIter = max(males$iter)

sel = seq(from=burn, to = maxIter, by=thin)

males = males %>% filter(iter %in% sel)
females = females %>% filter(iter %in% sel)

xlims = c(15,50)

## male source age
hdi50male = hdi(males$sourceAge, ci=0.5)
hdi50maleFixed = hdi(malesFixed$sourceAge, ci=0.5)

ggplot(males, aes(x=sourceAge)) +
  geom_density(aes(group=iter),
               color = alpha(wes_palette("Moonrise3")[1],0.2))+
  stat_density(bw=2,color = wes_palette("Darjeeling2")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=malesFixed, aes(x=sourceAge), 
               bw=2, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.06,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50male$CI_low, hdi50male$CI_high),
            size = 6, color = wes_palette("Darjeeling2")[2])+
  geom_text(x=40, y = 0.056,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50maleFixed$CI_low, hdi50maleFixed$CI_high),
            size = 6)+
  scale_x_continuous(limits = xlims)+
  labs(x='male source age', y='')+
  theme_bw(base_size = 14)


## female source age
hdi50female = hdi(females$sourceAge, ci=0.5)
hdi50femaleFixed = hdi(femalesFixed$sourceAge, ci=0.5)
ggplot(females, aes(x=sourceAge)) +
  geom_density(aes(group=iter), 
               color = alpha(wes_palette("Moonrise3")[2],0.2))+
  stat_density(bw=2.2,color = wes_palette("GrandBudapest1")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=femalesFixed, aes(x=sourceAge), 
               bw=2, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.06,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50female$CI_low, hdi50female$CI_high),
            size = 6, color = wes_palette("GrandBudapest1")[2])+
  geom_text(x=40, y = 0.056,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50femaleFixed$CI_low, hdi50femaleFixed$CI_high),
            size = 6)+
  labs(x='female source age', y='')+
  theme_bw(base_size = 14)

## 3.b. source/recipient age distribution for specific age group----

## e.g. I: young women (15-25)
source.males = read_csv('real_data_young_women_infection_male_source_age_samples.csv')
rec.males = read_csv('real_data_young_female_recipient_age_samples.csv')

source.males.fixed = read_csv('fixedThres_data_young_women_infection_male_source_age_samples.csv')
rec.males.fixed = read_csv('fixedThres_data_young_female_recipient_age_samples.csv')

source.males = source.males %>% filter(iter %in% sel)
rec.males = rec.males %>% filter(iter %in% sel)

source.males.fixed = source.males.fixed %>% filter(iter %in% sel)
rec.males.fixed = rec.males.fixed %>% filter(iter %in% sel)

ylims = c(0, 0.13)
xlims = c(15,50)

## male source age
hdi50male = hdi(source.males$sourceAge, ci=0.5)
hdi50male.fixed = hdi(source.males.fixed$sourceAge, ci=0.5)
ggplot(source.males, aes(x=sourceAge)) +
  geom_density(aes(group=iter), 
               color = alpha(wes_palette("Moonrise3")[1],0.2))+
  stat_density(bw=2,color = wes_palette("Darjeeling2")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=source.males.fixed, aes(x=sourceAge), 
               bw=2, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.10,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50male$CI_low, hdi50male$CI_high),
            size = 6, color = wes_palette("Darjeeling2")[2])+
  geom_text(x=40, y = 0.092,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50male.fixed$CI_low, hdi50male.fixed$CI_high),
            size = 6)+
  scale_y_continuous(limits = ylims)+
  scale_x_continuous(limits = xlims)+
  labs(x='male source age for adolescent and young women (15-24)', y='')+
  theme_bw(base_size = 14)

## male recipient age
hdi50male = hdi(rec.males$sourceAge, ci=0.5)
hdi50male.fixed = hdi(rec.males.fixed$sourceAge, ci=0.5)
ggplot(rec.males, aes(x=sourceAge)) +
  geom_density(aes(group=iter), 
               color = alpha(wes_palette("Moonrise3")[2],0.2))+
  stat_density(bw=1.8,color = wes_palette("GrandBudapest1")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=rec.males.fixed, aes(x=sourceAge), 
               bw=1.8, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.10,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50male$CI_low, hdi50male$CI_high),
            size = 6, color = wes_palette("GrandBudapest1")[2])+
  geom_text(x=40, y = 0.092,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50male.fixed$CI_low, hdi50male.fixed$CI_high),
            size = 6)+
  scale_y_continuous(limits = ylims)+
  scale_x_continuous(limits = xlims)+
  labs(x='male recipient age from dolescent and young women (15-24)', y='')+
  theme_bw(base_size = 14)


## e.g. II: young men (15-25)
source.females = read_csv('real_data_young_men_infection_female_source_age_samples.csv')
rec.females = read_csv('real_data_young_male_recipient_age_samples.csv')

source.females = source.females %>% filter(iter %in% sel)
rec.females = rec.females %>% filter(iter %in% sel)

source.females.fixed = read_csv('fixedThres_data_young_men_infection_female_source_age_samples.csv')
rec.females.fixed = read_csv('fixedThres_data_young_male_recipient_age_samples.csv')

source.females.fixed = source.females.fixed %>% filter(iter %in% sel)
rec.females.fixed = rec.females.fixed %>% filter(iter %in% sel)

ylims = c(0, 0.13)
xlims = c(15,50)

## female source age
hdi50female = hdi(source.females$sourceAge, ci=0.5)
hdi50femaleFixed = hdi(source.females.fixed$sourceAge, ci=0.5)
ggplot(source.females, aes(x=sourceAge)) +
  geom_density(aes(group=iter), 
               color = alpha(wes_palette("Moonrise3")[2],0.2))+
  stat_density(bw=2.2,color = wes_palette("GrandBudapest1")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=source.females.fixed, aes(x=sourceAge), 
               bw=2.2, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.10,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50female$CI_low, hdi50female$CI_high),
            size = 6, color = wes_palette("GrandBudapest1")[2])+
  geom_text(x=40, y = 0.092,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50femaleFixed$CI_low, hdi50femaleFixed$CI_high),
            size = 6)+
  scale_y_continuous(limits = ylims)+
  scale_x_continuous(limits = xlims)+
  labs(x='female source age for young men (15-25)', y='')+
  theme_bw(base_size = 14)

## female recipient age
hdi50female = hdi(rec.females$sourceAge, ci=0.5)
hdi50femaleFixed = hdi(rec.females.fixed$sourceAge, ci=0.5)
ggplot(rec.females, aes(x=sourceAge)) +
  geom_density(aes(group=iter), 
               color = alpha(wes_palette("Moonrise3")[1],0.2))+
  stat_density(bw=2,color = wes_palette("Darjeeling2")[2], 
               geom='line', position='identity', size = 1.5)+
  stat_density(data=rec.females.fixed, aes(x=sourceAge), 
               bw=2, color='grey30', geom='line', linetype = 2,
               position='identity', size = 1.5) +
  geom_text(x=40, y = 0.10,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50female$CI_low, hdi50female$CI_high),
            size = 6, color = wes_palette("Darjeeling2")[2])+
  geom_text(x=40, y = 0.092,
            label = sprintf('50%% HDI:[%.1f, %.1f]', 
                            hdi50femaleFixed$CI_low, hdi50femaleFixed$CI_high),
            size = 6)+
  scale_y_continuous(limits = ylims)+
  scale_x_continuous(limits = xlims)+
  labs(x='female recipient age from young men (15-25)', y='')+
  theme_bw(base_size = 14)



# 4. density of all estimated surface, with contours for HDIs-----
# Use Melodie's function to compute 2d credible regions for inference results (the density surfaces)

# her contour level function
getLevel <- function(x,y,z,prob=0.95) {
  dx <- diff(unique(x)[1:2])
  dy <- diff(unique(y)[1:2])
  sz <- sort(z)
  c1 <- cumsum(sz) * dx * dy
  approx(c1, sz, xout = 1 - prob)$y
}


## my code ------

library(wesanderson)

probs = c(0.5, 0.8, 0.9)

### 1. MF surface
MFdens = read_csv('MF_density_MAP.csv')

pal = wes_palette("Moonrise2", 3, type='continuous')
MFlevels = MFdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

p_MF = ggplot(MFdens,aes(x=x,y=y))+
  geom_raster(aes(fill = density)) +
  geom_contour(aes(z=density, col = ..level..), breaks = MFlevels$level) +
  geom_text(data = MFlabels, aes(label = prob_label, col = level), size = 4) +
  annotate(geom = 'text', x=21, y=47, 
           label = 'M->F transmission', size = 6)+
  #scale_color_viridis_c(option = 'C') +
  scale_color_gradientn(colors = pal)+
  scale_fill_gradient(low = 'white', 
                      high = wes_palette("Darjeeling2")[2]) +
  scale_x_continuous('male age', expand = c(0, 0)) + 
  scale_y_continuous('female age', expand = c(0, 0)) +
  guides(col="none", fill='none')+
  theme_bw(base_size = 14)

### 2. FM surface
FMdens = read_csv('FM_density_MAP.csv')

pal <- wes_palette("Zissou1", 3, type = "continuous")

FMlevels = FMdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

FMlabels = FMdens %>% 
  full_join(FMlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

p_FM = 
  ggplot(FMdens,aes(x=x,y=y))+
  geom_raster(aes(fill = density)) +
  geom_contour(aes(z=density, col = ..level..), breaks = FMlevels$level) +
  geom_text(data = FMlabels, aes(label = prob_label, col = level), size = 4) +
  annotate(geom = 'text', x=21, y=47, 
           label = 'F->M transmission', size = 6)+
  #scale_color_viridis_c(option = 'E') +
  scale_color_gradientn(colors = pal)+
  scale_fill_gradient(low = 'white', 
                      high = wes_palette("GrandBudapest1")[2]) +
  scale_x_continuous('male age', expand = c(0, 0)) + 
  scale_y_continuous('female age', expand = c(0, 0)) +
  guides(col="none", fill='none')+
  theme_bw(base_size = 14)





# 5 & 6. colored 2D points (with post.probs) and contour lines of HDIs---

# library(dplyr)
# library(ggplot2)
library(patchwork)
library(ggpubr)
#library(wesanderson)

# setwd('~/Documents/Research/HIV_transmission_flow/')
#dat = read_csv('Rakai_data_2022_with_type_freqs.csv') # data capped at L>=0.2

dat = read_csv('Rakai_data_Jan2022.csv') # raw data uncapped...
dat = dat %>% filter(!is.na(POSTERIOR_SCORE_LINKED),
                     !is.na(POSTERIOR_SCORE_MF),
                     !is.na(POSTERIOR_SCORE_FM))

# 1. Data demo plot for Section 2 ----------------

## 05/02/2022 update: color points differently
## darker color in Xi's paper
## lighter color: points also included in my paper
dat = dat %>% 
  mutate(paper = if_else(POSTERIOR_SCORE_LINKED > 0.6 & 
                           (POSTERIOR_SCORE_MF > 0.66 |POSTERIOR_SCORE_FM > 0.66),
                         'Xi', 'Bu'))
## a. age pairs scatter plot + histogram------
scatter1 = ggplot(dat, aes(x=MALE_AGE_AT_MID,
                           y= FEMALE_AGE_AT_MID,
                           color = paper)) +
  geom_point(size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_manual(values = c('grey70','grey30')) +
  labs(x='male age (mid-observation)', y = 'female age (mid-observation)') +
  geom_text(x=15, y=47, label = sprintf('previous: N=%s\nnow: N=%s', 
                                        sum(dat$paper=='Xi'),
                                        nrow(dat)),
            hjust = 0, size = 5)+
  theme_bw(base_size = 14) +
  theme(legend.position = 'none')

(hist_male = ggplot(dat, aes(x=MALE_AGE_AT_MID)) +
    geom_histogram(color = 'gray20', fill = 'gray90', bins = 25) +
    theme_void())

(hist_female = ggplot(dat, aes(x=FEMALE_AGE_AT_MID)) +
    geom_histogram(color = 'gray20', fill = 'gray90', bins = 25) +
    theme_void() + 
    coord_flip())


(age_scatter = hist_male + plot_spacer() +
    scatter1 + hist_female +
    plot_layout(ncol = 2, 
                nrow = 2, 
                widths = c(4, 1),
                heights = c(1, 4)))

## b. plots for linkage score distribution and direct score distribution
## histograms
dat = dat %>% 
  mutate(direction_type = case_when(POSTERIOR_SCORE_MF >= 0.66 ~ 'likely M->F',
                                    POSTERIOR_SCORE_MF <= 0.33 ~ 'likely F->M',
                                    TRUE ~ 'uncertatin'))
hist_link = ggplot(dat, aes(x=POSTERIOR_SCORE_LINKED)) +
  geom_histogram(color = 'gray20', fill = 'gray90', bins = 25) +
  geom_vline(xintercept = 0.6, linetype = 2, size = 1.5)+
  scale_x_continuous(limits = c(0,1),
                     labels = scales::percent)+
  scale_y_continuous(limits = c(0,50)) +
  labs(x='linkage score')+
  theme_bw(base_size = 14)

hist_direction = ggplot(dat, aes(x=POSTERIOR_SCORE_MF)) +
  geom_histogram(color = 'gray20', bins = 25, aes(fill=direction_type)) +
  scale_x_continuous(limits = c(0,1),
                     labels = scales::percent)+
  scale_y_continuous(limits = c(0,40)) +
  scale_fill_manual(values = c(wes_palette("Moonrise3")[2:1],'grey80')) +
  labs(x='M->F direction score',fill='')+
  theme_bw(base_size = 14) +
  theme(legend.position = 'bottom',
        legend.text = element_text(size=10),
        legend.key.size = unit(0.4, 'cm'),
        legend.margin = margin(t = -5, r = 0, b = 0, l = 0, unit = "pt"))

ggarrange(age_scatter, 
          ggarrange(hist_link, hist_direction, nrow = 2, labels = c('B','C')),
          ncol = 2,
          labels = 'A')


# 2. section 5: data colored with scores --------
dat2 = read_csv('Rakai_data_2022_with_type_freqs.csv') # using processed data in analysis

dat2Xi = dat2 %>% 
  filter(POSTERIOR_SCORE_LINKED >= 0.6) %>%
  mutate(direction = if_else(POSTERIOR_SCORE_MF >= 0.5, 
                             'M->F', 'F->M'))

## a. age pairs with pre-classification------
preMF  = ggplot(dat2Xi %>% 
                  filter(direction == 'M->F'), 
                aes(x=MALE_AGE_AT_MID,
                    y= FEMALE_AGE_AT_MID)) +
  geom_point(color = "gray30", size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  labs(x='male age', y = 'female age',
       title = 'With pre-classification') +
  theme_bw(base_size = 14)+
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 20))

preFM = ggplot(dat2Xi %>% 
                 filter(direction == 'F->M'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID)) +
  geom_point(color = "gray30", size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  labs(x='male age', y = 'female age') +
  theme_bw(base_size = 14)

## b. all age pairs with coloring ------
## color by posterior probs------
allMF = ggplot(dat2,
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_MF)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_distiller(type = "seq",
                        direction = 1,
                        palette = "Greys",
                        limits = c(0,1.0),
                        breaks = c(0,0.25,0.5,0.75,1.0),
                        labels = scales::percent(c(0,0.25,0.5,0.75,1.0)))+
  labs(x='male age', y = 'female age', 
       color='posterior\nM->F\nprobability',
       title = 'Including all data') +
  theme_bw(base_size = 14)+
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 20))

allFM = ggplot(dat2,
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_FM)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_distiller(type = "seq",
                        direction = 1,
                        palette = "Greys",
                        limits = c(0,1.0),
                        breaks = c(0,0.25,0.5,0.75,1.0),
                        labels = scales::percent(c(0,0.25,0.5,0.75,1.0)))+
  labs(x='male age', y = 'female age', 
       color='posterior\nF->M\nprobability') +
  theme_bw(base_size = 14)

ggarrange(ggarrange(preMF+rremove("xlab")+rremove("x.text")+rremove('x.ticks'),
                    allMF+rremove("xylab")+rremove("axis.text")+rremove('ticks'), 
                    ncol=2,
                    widths = c(3.3,3.8)),
          ggarrange(preFM, 
                    allFM+rremove("ylab")+rremove("y.text")+rremove('y.ticks'), 
                    ncol=2,
                    widths = c(3.3,3.8)),
          nrow = 2,
          heights = c(3,3.2),
          labels = c('A','B'))

## 3. presentation illustration ------
## used for making presentation slides
## contrasting fixed threshold approach and flexible approach
dat2 = read_csv('Rakai_data_2022_with_type_freqs.csv') # using processed data in analysis

dat2Xi = dat2 %>% 
  filter(POSTERIOR_SCORE_LINKED >= 0.6) %>%
  mutate(direction = case_when(POSTERIOR_SCORE_MF >= 0.66 ~ 'M->F',
                               POSTERIOR_SCORE_MF <= 0.33 ~ 'F->M',
                               TRUE ~ 'none'))

dat2fixed = dat2 %>%
  mutate(direction = case_when((POSTERIOR_SCORE_MF >= 0.5) & (POSTERIOR_SCORE_LINKED >= 0.6)  ~ 'M->F',
                               (POSTERIOR_SCORE_MF < 0.5) & (POSTERIOR_SCORE_LINKED >= 0.6) ~ 'F->M',
                               TRUE ~ 'none'))

## (1) Xi's model
## points
preMF  = ggplot(dat2Xi %>% 
                  filter(direction == 'M->F'), 
                aes(x=MALE_AGE_AT_MID,
                    y= FEMALE_AGE_AT_MID)) +
  geom_point(color = "gray30", size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  geom_text(x=20, y=48, label = sprintf('N=%s', sum(dat2Xi$direction=='M->F')),
            size = 12)+
  labs(x='male age', y = 'female age') +
  theme_bw(base_size = 14)


## discretized spatial tiles
get_age_bin <- function(x, breaks = seq(15,50,by=1)){
  num_bin = length(breaks) - 1
  res = character(length(x))
  for(i in 1:num_bin){
    lb = breaks[i]
    ub = breaks[i+1]
    res[x>lb & x<ub] = paste0(as.character(lb),'-',as.character(ub))
  }
  res
}
dat2XiMF = dat2Xi %>% 
  filter(direction == 'M->F') %>%
  mutate(male_age_group = get_age_bin(MALE_AGE_AT_MID),
         female_age_group = get_age_bin(FEMALE_AGE_AT_MID))

age_group_counts = dat2XiMF %>% 
  group_by(male_age_group, female_age_group) %>%
  summarise(num_points = n())

preMFtiles = ggplot(age_group_counts,
                    aes(x=male_age_group,
                        y=female_age_group,
                        fill = num_points)) +
  geom_tile() +
  scale_fill_gradient(low = alpha(wes_palette("Moonrise3")[1], 0.4), 
                      high = wes_palette("Darjeeling2")[2]) +
  labs(x='male age', y = 'female age') +
  theme_bw(base_size = 14) +
  theme(legend.position = 'none',
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


## (2) My model
## points
allMF = ggplot(dat2,
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_MF)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_gradient(low = 'grey85', high='black')+
  # scale_color_distiller(type = "seq",
  #                       direction = 1,
  #                       palette = "Greys",
  #                       limits = c(0,1.0),
  #                       breaks = c(0,0.25,0.5,0.75,1.0),
  #                       labels = scales::percent(c(0,0.25,0.5,0.75,1.0)))+
  geom_text(x=20, y=48, label = sprintf('N=%s', nrow(dat2)), size = 12)+
  labs(x='male age', y = 'female age', 
       color='posterior\nM->F\nprobability') +
  theme_bw(base_size = 14)+
  theme(legend.position = 'none')

## continuous 2d density
allMFdens = ggplot(dat2 %>% filter(freq_MF > 0.6),
                   aes(x=MALE_AGE_AT_MID,
                       y= FEMALE_AGE_AT_MID)) +
  stat_density_2d(geom = "polygon", contour = TRUE, 
                  aes(fill = after_stat(level)), 
                  colour = "gray60",bins = 10) +
  # scale_fill_distiller(palette = "Blues", direction = 1) +
  scale_fill_gradient(low = 'white', 
                      high = wes_palette("Darjeeling2")[2]) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  #geom_text(x=18, y=48, label = sprintf('N=%s', nrow(dat2)))+
  labs(x='male age', y = 'female age') +
  theme_bw(base_size = 14)+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())


### spatial points colored with posterior type probs-------
## the figure used in Section 5 of the paper

## (a) Xi paper classification
preMFcolored  = ggplot(dat2Xi %>% 
                         filter(direction == 'M->F'), 
                       aes(x=MALE_AGE_AT_MID,
                           y= FEMALE_AGE_AT_MID)) +
  geom_point(color = wes_palette("Moonrise3")[1], size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  geom_text(x=18, y=48, label = sprintf('N=%s', sum(dat2Xi$direction=='M->F')), size = 6)+
  labs(x='', y = 'female age', title = 'M->F transmission') +
  theme_bw(base_size = 14)+
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 22))

preFMcolored  = ggplot(dat2Xi %>% 
                         filter(direction == 'F->M'), 
                       aes(x=MALE_AGE_AT_MID,
                           y= FEMALE_AGE_AT_MID)) +
  geom_point(color = wes_palette("Moonrise3")[2], size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  geom_text(x=18, y=48, label = sprintf('N=%s', sum(dat2Xi$direction=='F->M')), size = 6)+
  labs(x='', y = '', title = 'F->M transmission') +
  theme_bw(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 22))

pre0colored = ggplot(dat2Xi %>% 
                       filter(direction == 'none'), 
                     aes(x=MALE_AGE_AT_MID,
                         y= FEMALE_AGE_AT_MID)) +
  geom_point(color = 'grey50', size = 1.8) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  geom_text(x=18, y=48, label = sprintf('N=%s', sum(dat2Xi$direction=='none')), size = 6)+
  labs(x='', y = '', title = 'no transmission link') +
  theme_bw(base_size = 14)+
  theme(plot.title = element_text(hjust = 0.5,
                                  size = 22))

## (b) my method's flexible classification
# FM Pink: wes_palette("GrandBudapest1")[2]
# MF Blue: wes_palette("Darjeeling2")[2]

allMFcolored = ggplot(dat2,
                      aes(x=MALE_AGE_AT_MID,
                          y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_MF)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_gradient(low = 'white', high=wes_palette("Darjeeling2")[2])+
  geom_text(x=18, y=48, label = sprintf('N=%s', nrow(dat2)), size = 6)+
  labs(x='male age', y = 'female age', 
       color='posterior\nM->F\nprobability') +
  theme_bw(base_size = 14)+
  theme(legend.position = 'none')


allFMcolored = ggplot(dat2,
                      aes(x=MALE_AGE_AT_MID,
                          y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_FM)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_gradient(low = 'white', high=wes_palette("GrandBudapest1")[2])+
  geom_text(x=18, y=48, label = sprintf('N=%s', nrow(dat2)), size = 6)+
  labs(x='male age', y = '', 
       color='posterior\nF->M\nprobability') +
  theme_bw(base_size = 14)+
  theme(legend.position = 'none')

all0colored = ggplot(dat2,
                     aes(x=MALE_AGE_AT_MID,
                         y= FEMALE_AGE_AT_MID)) +
  geom_point(size = 1.8, aes(color = freq_0)) +
  scale_x_continuous(limits = c(15,50)) +
  scale_y_continuous(limits = c(15,50)) +
  scale_color_gradient(low = 'white', high='black')+
  geom_text(x=18, y=48, label = sprintf('N=%s', nrow(dat2)), size = 6)+
  labs(x='male age', y = '', 
       color='') +
  theme_bw(base_size = 14)+
  theme(legend.position = 'none')



ggarrange(ggarrange(preMFcolored,
                    preFMcolored,
                    pre0colored,
                    ncol=3),
          ggarrange(allMFcolored, 
                    allFMcolored,
                    all0colored,
                    ncol=3),
          nrow = 2,
          #heights = c(3,3.2),
          labels = c('A','B'))


### colored points with spatial density contour lines as well------
## (June 11 updated version)
## (June 20 updates again)

## July 19 2022: add diagonal lines for more clear comparison on symmetry

## the level function shared by Melodie M. 
getLevel <- function(x,y,z,prob=0.95) {
  dx <- diff(unique(x)[1:2])
  dy <- diff(unique(y)[1:2])
  sz <- sort(z)
  c1 <- cumsum(sz) * dx * dy
  approx(c1, sz, xout = 1 - prob)$y
}

### try out additional color scales (for the points and the contours)
library(ggnewscale)

## look at those probs levels
probs = c(0.5, 0.8, 0.9)

# (a). MF surface first...
## the data for MF densities from full model
MFdens = read_csv('MF_density_MAP.csv')

pal = wes_palette("Moonrise2", 3, type='continuous')
MFlevels = MFdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

## try some different text label positions here
## to avoid text-point overlap
MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  arrange(diffs) %>%
  slice(40) %>% # probably need to toggle this line and the next to find a good one
  filter(x == max(x)) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

# reproduce "allMFcolored" but with contour lines of HDIs
(
  allMFcolored_contours = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2, 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID,
                   color = freq_MF)) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    scale_color_gradient(low = 'white', high=wes_palette("Darjeeling2")[2])+
    geom_text(data = dat2, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47,  
              label = sprintf('Model:\nN=%s', nrow(dat2)), size = 6)+
    new_scale("color") +
    geom_contour(data = MFdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = MFlevels$level) +
    geom_text(data = MFlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', 
         color='posterior\nM->F\nprobability') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none')
)

# (b). FM surface
FMdens = read_csv('FM_density_MAP.csv')

pal <- wes_palette("Zissou1", 3, type = "continuous")

FMlevels = FMdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

FMlabels = FMdens %>% 
  full_join(FMlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

## reproduce "allFMcolored" but with contour lines of HDIs
(
  allFMcolored_contours = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2, aes(color = freq_FM, 
                                x=MALE_AGE_AT_MID,
                                y= FEMALE_AGE_AT_MID), 
               size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    scale_color_gradient(low = 'white', high=wes_palette("GrandBudapest1")[2])+
    geom_text(data = dat2, aes(x=MALE_AGE_AT_MID,
                               y= FEMALE_AGE_AT_MID), 
              x=19, y=47,  
              label = sprintf('Model:\nN=%s', nrow(dat2)), size = 6)+
    new_scale("color") +
    geom_contour(data = FMdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = FMlevels$level) +
    geom_text(data = FMlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', 
         color='posterior\nF->M\nprobability') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none')
)


# (c). "none" surface
#nonedens = read_csv('0_density_MAP.csv')
nonedens = read_csv('0_density_Mean.csv')

#pal <- wes_palette("Zissou1", 3, type = "continuous")

nonelevels = nonedens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

nonelabels = nonedens %>% 
  full_join(nonelevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))


## reproduce "all0colored" but with contour lines of HDIs
(
  allnonecolored_contours = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2, aes(color = freq_0, 
                                x=MALE_AGE_AT_MID,
                                y= FEMALE_AGE_AT_MID), 
               size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    scale_color_gradient(low = 'white', high='black')+
    geom_text(data = dat2, aes(x=MALE_AGE_AT_MID,
                               y= FEMALE_AGE_AT_MID), 
              x=19, y=47, 
              label = sprintf('Model:\nN=%s', nrow(dat2)), size = 6)+
    new_scale("color") +
    geom_contour(data = nonedens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = nonelevels$level) +
    geom_text(data = nonelabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', 
         color='posterior\nnone\nprobability') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none')
)


## (ii) then also plot the fixed analysis results----

## (a) MF surface

## the data for MF densities from partial model
MFdens = read_csv('fixThres_MF_density_MAP.csv')

pal = wes_palette("Moonrise2", 3, type='continuous')
MFlevels = MFdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

## try some different text label positions here
## to avoid text-point overlap
MFlabels = MFdens %>%
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  arrange(diffs) %>%
  slice(40) %>% # probably need to toggle this line and the next to find a good one
  filter(x == max(x)) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

(
  preMFcolored  = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2fixed %>% 
                 filter(direction == 'M->F'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID),
               color = wes_palette("Moonrise3")[1], size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    geom_text(data = dat2fixed, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47, 
              label = sprintf('Fixed:\nN=%s', sum(dat2fixed$direction=='M->F')), size = 6)+
    geom_contour(data = MFdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = MFlevels$level) +
    geom_text(data = MFlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', title = 'M->F transmissions') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5,
                                    size = 22))
)


## (b) FM surface
FMdens = read_csv('fixThres_FM_density_MAP.csv')

pal <- wes_palette("Zissou1", 3, type = "continuous")

FMlevels = FMdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

FMlabels = FMdens %>% 
  full_join(FMlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))

(
  preFMcolored  = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2fixed %>% 
                 filter(direction == 'F->M'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID),
               color = wes_palette("Moonrise3")[2], size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    geom_text(data = dat2fixed, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47,  
              label = sprintf('Fixed:\nN=%s', sum(dat2fixed$direction=='F->M')), size = 6)+
    geom_contour(data = FMdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = FMlevels$level) +
    geom_text(data = FMlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', title = 'F->M transmissions') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5,
                                    size = 22))
)

## (c) none surface
nonedens = read_csv('fixThres_0_density_MAP.csv')

#pal <- wes_palette("Zissou1", 3, type = "continuous")

nonelevels = nonedens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

nonelabels = nonedens %>% 
  full_join(nonelevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))


(
  prenonecolored  = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2fixed %>% 
                 filter(direction == 'none'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID),
               color = 'gray50', size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    geom_text(data = dat2fixed, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47, 
              label = sprintf('Fixed:\nN=%s', sum(dat2fixed$direction=='none')), 
              size = 6)+
    geom_contour(data = nonedens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = nonelevels$level) +
    geom_text(data = nonelabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', title = 'None/unknown') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5,
                                    size = 22))
)


ggarrange(ggarrange(preMFcolored,
                    preFMcolored,
                    prenonecolored,
                    ncol=3),
          ggarrange(allMFcolored_contours, 
                    allFMcolored_contours,
                    allnonecolored_contours,
                    ncol=3),
          nrow = 2,
          heights = c(3.2,3),
          labels = c('A','B'))


# 11/06/2022
# presentation examples with MF surface only for better visualization----

## (1) without contour lines----
## MF using fixed model
(
  preMFcolored2  = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2fixed %>% 
                 filter(direction == 'M->F'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID),
               color = wes_palette("Darjeeling2")[2], size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    geom_text(data = dat2fixed, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47, 
              label = sprintf('Fixed:\nN=%s', sum(dat2fixed$direction=='M->F')), size = 6)+
    # geom_contour(data = MFdens,
    #              aes(x=x,y=y, z=density, col = ..level..),
    #              breaks = MFlevels$level) +
    # geom_text(data = MFlabels,
    #           aes(x=x, y=y, label = prob_label, col = level),
    #           size = 4) +
    #scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5,
                                    size = 22))
)


## MF using full model
(
  allMFcolored2 = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2, 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID,
                   color = freq_MF)) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    scale_color_gradient(low = 'white', high=wes_palette("Darjeeling2")[2])+
    geom_text(data = dat2, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47,  
              label = sprintf('Model:\nN=%s', nrow(dat2)), size = 6)+
    #new_scale("color") +
    # geom_contour(data = MFdens,
    #              aes(x=x,y=y, z=density, col = ..level..),
    #              breaks = MFlevels$level) +
    # geom_text(data = MFlabels,
    #           aes(x=x, y=y, label = prob_label, col = level),
    #           size = 4) +
    #scale_color_gradientn(colors = pal)+
    #scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', 
         color='posterior\nM->F\nprobability') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'right')
)


ggarrange(preMFcolored2, allMFcolored2,
          ncol = 2,
          widths = c(3, 4),
          labels = c('',''))

## (2) adding density contour lines
## MF using fixed model
## the data for MF densities from partial model
MFdens = read_csv('fixThres_MF_density_MAP.csv')

pal = wes_palette("Moonrise2", 3, type='continuous')
MFlevels = MFdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))
(
  preMFcolored2_contour  = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2fixed %>% 
                 filter(direction == 'M->F'), 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID),
               color = wes_palette("Darjeeling2")[2], size = 1.8) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    geom_text(data = dat2fixed, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47, 
              label = sprintf('Fixed:\nN=%s', sum(dat2fixed$direction=='M->F')), size = 6)+
    geom_contour(data = MFdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = MFlevels$level) +
    geom_text(data = MFlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none',
          plot.title = element_text(hjust = 0.5,
                                    size = 22))
)


## MF using full model
MFdens = read_csv('MF_density_MAP.csv')

pal = wes_palette("Moonrise2", 3, type='continuous')
MFlevels = MFdens %>% 
  summarise(level = getLevel(x,y,density, prob = probs)) %>%
  mutate(prob = probs)

MFlabels = MFdens %>% 
  full_join(MFlevels,by = character()) %>%
  mutate(diffs = abs(density - level)) %>%
  group_by(prob) %>%
  filter(diffs == min(diffs)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(prob_label = sprintf('%.0f%%', prob * 100))
(
  allMFcolored2_contour = 
    ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color='gray70') +
    geom_point(data = dat2, 
               aes(x=MALE_AGE_AT_MID,
                   y= FEMALE_AGE_AT_MID,
                   color = freq_MF)) +
    scale_x_continuous(limits = c(15,50)) +
    scale_y_continuous(limits = c(15,50)) +
    scale_color_gradient(low = 'white', high=wes_palette("Darjeeling2")[2])+
    geom_text(data = dat2, 
              aes(x=MALE_AGE_AT_MID,
                  y= FEMALE_AGE_AT_MID),
              x=19, y=47,  
              label = sprintf('Model:\nN=%s', nrow(dat2)), size = 6)+
    new_scale("color") +
    geom_contour(data = MFdens,
                 aes(x=x,y=y, z=density, col = ..level..),
                 breaks = MFlevels$level) +
    geom_text(data = MFlabels,
              aes(x=x, y=y, label = prob_label, col = level),
              size = 4) +
    scale_color_gradientn(colors = pal)+
    scale_color_gradient(low = 'gray60', high = 'gray10')+
    labs(x='male age', y = 'female age', 
         color='posterior\nM->F\nprobability') +
    theme_bw(base_size = 14)+
    theme(legend.position = 'none')
)


ggarrange(preMFcolored2_contour, allMFcolored2_contour,
          ncol = 2,
          widths = c(3, 3),
          labels = c('',''))
