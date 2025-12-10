# Conflict in Ethiopia: Before and After Abiy Ahmed


## Quantitative Analysis


## Executive Summary

This analysis examines conflict trends in Ethiopia before and after Abiy Ahmed assumed power on April 02, 2018. Using data from the Armed Conflict Location & Event Data Project (ACLED), we compare conflict events and fatalities across two periods: pre-Abiy (2008-2018) and post-Abiy (2018-2025). The analysis reveals significant changes in both the intensity and nature of conflict following the transition.

## Introduction

The Armed Conflict Location & Event Data Project (ACLED) provides comprehensive data on political violence and protest events worldwide. This analysis leverages ACLED data to examine how conflict patterns in Ethiopia changed following the appointment of Abiy Ahmed as Prime Minister on April 02, 2018. By comparing equal-duration periods before and after this transition, we aim to provide a quantitative assessment of conflict trends while acknowledging the descriptive (non-causal) nature of this comparison.

## Methodology

To ensure fair temporal comparison, we balanced the pre and post-Abiy periods by preserving all post-Abiy data and matching the pre-Abiy window to the same duration. This time-based balancing approach ensures equal time periods for comparison, regardless of event counts. The pre-Abiy period spans from August 05, 2011 to April 02, 2018, and the post-Abiy period spans from April 02, 2018 to December 10, 2024, with the cutoff date of April 02, 2018 serving as the dividing point. All rates are calculated as events or fatalities per month to account for period length differences. Administrative boundaries at Level 1 (regions) are used for spatial analysis, and an interrupted time-series (ITS) regression model is employed to assess statistical significance of observed changes.

## Event and Fatality Trends

The data reveal substantial increases in conflict activity following Abiy Ahmed's assumption of power. As shown in Table 1 and Figure 1, the average number of conflict events per month increased from 38.2 to 165.1, representing a 332.1% increase. Similarly, fatalities per month rose from 137.2 to 539.9, a 293.5% increase.

![Table 1: Summary of Monthly Conflict Events and Fatalities](tables/tbl01_pre_post_overall.png)

![Figure 1: Monthly Conflict Events](figures/fig01_monthly_events.png)




Total events increased from 3,209 in the pre-Abiy period to 9,905 in the post-Abiy period, while total fatalities rose from 11,525 to 32,393. The mean fatalities per event remained relatively stable, changing from 3.59 to 3.27 fatalities per event. These trends are visualized in Figure 1 (monthly events) and Figure 2 (monthly fatalities), which show a clear upward trajectory following the cutoff date.

![Figure 2: Monthly Conflict Fatalities](figures/fig02_monthly_fatalities.png)



## Event Type Composition Changes

The composition of conflict events shifted significantly between periods, as illustrated in Table 2 and Figure 3. The most notable changes include:

![Table 2: Event Type Distribution](tables/tbl02_event_type_distribution.png)

![Figure 3: Event Type Composition](figures/fig03_event_type_composition.png)




**Event Types with Largest Increases:**

1. **Battles**: Increased from 28.9% to 47.8% of total events (+18.9% percentage points; see Table 2).

2. **Violence against civilians**: Increased from 14.8% to 25.2% of total events (+10.4% percentage points; see Table 2).

3. **Strategic developments**: Increased from 3.1% to 9.6% of total events (+6.5% percentage points; see Table 2).


**Event Types with Largest Decreases:**

1. **Protests**: Decreased from 44.1% to 10.6% of total events (-33.5% percentage points).

2. **Riots**: Decreased from 7.1% to 2.6% of total events (-4.5% percentage points).


These shifts reflect changes in the nature of conflict, with certain types of violence becoming more or less prominent in the post-Abiy period.

## Regional Patterns

Conflict intensity changed unevenly across Ethiopia's regions, as shown in Table 3 and Figures 4-5. The spatial distribution reveals both increases and decreases in conflict activity:

![Table 3: Regional Averages](tables/tbl03_regional_averages_admin1.png)

![Figure 4: Regional Distribution](figures/fig04_regional_distribution_admin1.png)

![Figure 5: Regional Change](figures/fig05_regional_change_admin1.png)




**Regions with Largest Increases:**

1. **Amhara**: Increased by 35.79 events per month (see Table 3 and Figure 4 for regional distribution).

2. **Tigray**: Increased by 15.16 events per month (see Table 3 and Figure 4 for regional distribution).

3. **Oromia**: Increased by 10.09 events per month (see Table 3 and Figure 4 for regional distribution).

4. **Afar**: Increased by 3.26 events per month (see Table 3 and Figure 4 for regional distribution).

5. **SNNP**: Increased by 2.94 events per month (see Table 3 and Figure 4 for regional distribution).


**Regions with Largest Decreases:**

1. **Somali**: Decreased by 1.26 events per month.


The regional change map (Figure 5) provides a visual representation of these spatial patterns, highlighting areas of increased and decreased conflict intensity.

## Statistical Significance

An interrupted time-series (ITS) regression model was fitted to assess the statistical significance of observed changes. The model (Table 4, Figure 6) reveals significant effects: 
the immediate level change following the cutoff date is -191.15 events per month, 
and the trend change (interaction term) is 1.68 events per month. 
The model explains 60.5% of the variance in monthly event counts (R² = 0.605). These results confirm that the observed increases in conflict are statistically significant beyond what would be expected from pre-existing trends alone.

![Table 4: ITS Regression Coefficients](tables/tbl04_its_coefficients.png)

![Figure 6: ITS Model Fit](figures/fig06_its_model_fit.png)



## Conclusion

This quantitative analysis reveals substantial increases in conflict activity in Ethiopia following Abiy Ahmed's assumption of power in 2018. Both the frequency and intensity of conflict events increased significantly, with notable shifts in event type composition and regional distribution. While these findings are descriptive and do not establish causality, they provide a comprehensive quantitative assessment of conflict trends during this period. The data suggest that the post-Abiy period has been marked by heightened conflict activity across multiple dimensions, requiring further qualitative and contextual analysis to fully understand the underlying dynamics.




## Data Limitations and Caveats

Several important limitations should be considered when interpreting these findings. First, ACLED data collection relies on media reports, government sources, and other publicly available information, which may introduce reporting biases. Events in remote or less-accessible areas may be under-reported, and media attention may vary over time and across regions. Additionally, the quality and availability of source information may differ between the pre and post-Abiy periods, potentially affecting comparability.


To address differences in period length, we employed time-based balancing that preserves all post-Abiy data and matches the pre-Abiy window to the same duration. All rates are calculated as events or fatalities per month to normalize for period length. However, this approach does not account for potential seasonal patterns or long-term trends that may have existed independently of the political transition. The interrupted time-series model attempts to control for pre-existing trends, but residual confounding factors may remain.


Most importantly, this analysis is descriptive rather than causal. The observed changes in conflict patterns cannot be definitively attributed to Abiy Ahmed's leadership or policies alone. Multiple factors—including economic conditions, regional dynamics, international relations, and historical legacies—likely contribute to conflict trends. The pre/post comparison provides a quantitative assessment of changes but does not establish causality. Future research combining quantitative analysis with qualitative investigation would be valuable for understanding the mechanisms underlying these trends.


Finally, while ACLED provides comprehensive coverage, data quality and completeness may vary across event types and regions. Some conflict events may be missed entirely, and fatality counts are often estimates with varying degrees of uncertainty. Readers should interpret the findings with these limitations in mind and consider them as part of a broader body of evidence on conflict in Ethiopia.
