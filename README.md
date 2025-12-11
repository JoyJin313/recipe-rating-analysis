# Do low-fat recipes receive different ratings compared to high-fat recipes?

**Author**: Leyi Jin

## Overview

This data science project conducted at UCSD is about exploring whether low-fat recipies receive different ratings compared to high-fat recipes.

## Introduction

Food is an essential part of our daily lives, and cooking is an activity that brings both comfort and creativity to many people. As health consciousness rises, nutritional components such as fat have become a central topic when people choose what to cook and eat. While dietary fat is necessary for our bodies, excessive fat intake has been linked to a variety of health issues, including obesity and cardiovascular diseases. According to the Centers for Disease Control and Prevention, about 40% of U.S. adults are classified as obese, and heart disease remains the leading cause of death nationwide. With these health concerns in mind, we want to investigate whether people rate low-fat and high-fat recipes differently. Specifically, we wonder if users tend to give lower ratings to high-fat recipes because they perceive them as unhealthy. To do so, we analyze two datasets consisting of recipes and ratings posted since 2008 on [food.com](https://www.food.com).

The first dataset, recipe, contains 83,782 rows, each representing a unique recipe, with 10 columns recording the following information:
| Column           | Description                                                                                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`           | Recipe name                                                                                                                                                                                              |
| `id`             | Recipe ID                                                                                                                                                                                                |
| `minutes`        | Minutes required to prepare the recipe                                                                                                                                                                   |
| `contributor_id` | User ID who submitted the recipe                                                                                                                                                                         |
| `submitted`      | Date the recipe was submitted                                                                                                                                                                            |
| `tags`           | Food.com tags associated with the recipe                                                                                                                                                                 |
| `nutrition`      | Nutrition vector in the form:<br>`[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`<br>where PDV stands for Percentage of Daily Value |
| `n_steps`        | Number of steps in the recipe                                                                                                                                                                            |
| `steps`          | Text description of recipe preparation steps                                                                                                                                                             |
| `description`    | User-provided recipe description                                                                                                                                                                         |
| `ingredients`    | Text of the recipe ingredients                                                                                                                                                                           |
| `n_ingredients`  | Number of ingredients                                                                                                                                                                                    |

The second dataset, interactions, contains 731,927 rows, each recording one user’s interaction with a recipe. It includes the following columns:
| Column      | Description                      |
| ----------- | -------------------------------- |
| `user_id`   | User ID                          |
| `recipe_id` | Recipe ID                        |
| `date`      | Date of interaction              |
| `rating`    | Rating assigned by the user      |
| `review`    | Text review provided by the user |

Given these datasets, we investigate whether people rate low-fat and high-fat recipes on the same scale. To support this analysis, we first separated the values in the `nutrition` column into their individual components, including calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV). Using these expanded variables, we then computed a new feature, `prop_fat`, defined as the proportion of total fat PDV relative to the total calories for each recipe. The resulting `prop_fat` values range from 0.0000 to 0.1871, with a dataset-wide mean of approximately **0.06829**. 

Based on this mean value, we classified recipes as **high-fat** if their `prop_fat` is above 0.06829 and **low-fat** if their value falls below this threshold. This classification results in **43,249 high-fat recipes** and **40,533 low-fat recipes**, providing a balanced basis for comparison.

The most relevant variables for exploring our question are therefore `total_fat_PDV`, `sat_fat_PDV`, `prop_fat`, `rating`, and the average rating of each recipe. By examining whether user ratings systematically differ between low-fat and high-fat recipes, we aim to gain insight into users’ preferences regarding fat content. Such findings may help contributors on Food.com refine their recipe designs to better align with public tastes and health considerations, and may also motivate future investigations into how nutritional awareness shapes user behavior.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
To prepare the datasets for analysis, we performed several data cleaning steps to combine information from the recipes and interactions tables and to compute an average rating for each recipe.

- Merging the Recipes and Interactions Datasets
- Checking and Validating Data Types
- Replacing Ratings of 0 with NaN
  - In the merged dataset, we replaced all ratings of **0** with `np.nan`. A rating of 0 is not a valid Food.com user rating (the valid range is 1–5). A value of 0 usually indicates a placeholder, a missing rating, or a user who submitted a review but did not record a numerical score. Treating these values as missing (`NaN`) is a reasonable and necessary step to avoid artificially lowering the average rating for a recipe.
- Computing the Average Rating per Recipe
  - After cleaning the ratings, we computed the **average rating for each recipe** by grouping the merged dataset on `recipe_id` and taking the mean of the `rating` column. This produces a Series indexed by recipe ID.
- Adding the Average Rating Back to the Recipes Dataset
  - Finally, we merged this Series of average ratings back into the original recipes dataset. Each recipe now contains a new variable, `avg_rating`, which reflects the mean user rating across all reviews. This enriched dataset will be used for the remainder of the analysis.
 
**Result**

| Column Name        | Data Type |
|--------------------|-----------|
| name               | object    |
| id                 | int64     |
| minutes            | int64     |
| contributor_id     | int64     |
| submitted          | object    |
| tags               | object    |
| nutrition          | object    |
| n_steps            | int64     |
| steps              | object    |
| description        | object    |
| ingredients        | object    |
| n_ingredients      | int64     |
| calories           | float64   |
| total_fat_PDV      | float64   |
| sugar_PDV          | float64   |
| sodium_PDV         | float64   |
| protein_PDV        | float64   |
| sat_fat_PDV        | float64   |
| carb_PDV           | float64   |
| prop_fat           | float64   |
| fat_category       | object    |
| avg_rating         | float64   |

### Univariate Analysis

Before exploring relationships between variables, we first examine the distributions of several key variables individually. Since our main research question concerns the impact of fat content on recipe ratings, we focus on nutritional and rating-related variables, including `prop_fat`, `total_fat_PDV`, and `avg_rating`.

#### Distribution of `prop_fat`

The variable `prop_fat` represents the proportion of total fat (PDV) relative to the total calories of each recipe. Understanding the distribution of `prop_fat` helps us evaluate whether fat levels vary widely across recipes or cluster around certain values. A histogram and a boxplot are shown below.

<iframe
  src="asset/Univariate_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="asset/Univariate_boxplot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

From the histogram of `prop_fat`, we observe that the distribution of fat proportion is right-skewed. Most recipes fall within the range of **0.04 to 0.10**, with the center of the distribution located near the mean value of approximately 0.068. There is also a noticeable spike near 0, indicating that a subset of recipes contains very little or almost no fat.

The boxplot further supports these observations. The median lies around **0.07**, and the interquartile range (IQR) spans roughly from 0.05 to 0.10, showing that the majority of recipes cluster tightly in this low-fat region. A few points appear on the far right side of the boxplot, representing high-fat outliers with fat proportions greater than **0.15**.

Overall, the distribution of `prop_fat` is fairly concentrated and skewed toward lower values. Most recipes in the dataset are low-fat or moderately low-fat, while high-fat recipes make up only a small portion. This pattern provides useful context for later comparisons of how low-fat versus high-fat recipes are rated by users.

### Bivariate Analysis

To better understand the relationship between fat proportion and user ratings, we examine how `prop_fat` is associated with the average rating a recipe receives. Since our research question asks whether low-fat and high-fat recipes are rated differently, it is important to visualize how these variables interact.

We produce two sets of plots:

1. A **scatter plot** of `prop_fat` vs. `avg_rating` to illustrate their overall relationship.  
2. A **boxplot of average ratings**, conditioned on whether a recipe is categorized as low-fat or high-fat.  

Together, these visualizations help us identify patterns that may motivate a meaningful hypothesis test.

<iframe
  src="asset/Bivariate_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The scatter plot of `prop_fat` versus `avg_rating` shows that average ratings are heavily clustered between 4.0 and 5.0 across all fat proportions. This indicates that most recipes, regardless of their fat content, tend to receive high ratings on Food.com. There is no strong visible trend suggesting that higher or lower fat proportion consistently influences the average rating. Instead, the plot shows a wide vertical band at each fat level, reflecting the fact that user ratings are concentrated at the upper end of the scale.

<iframe
  src="asset/Bivariate_boxplot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


The boxplot comparing average ratings between high-fat and low-fat recipes provides a clearer side-by-side comparison. Both categories have very similar medians (close to 5.0) and comparable overall distributions. While there are more low-rating outliers in the low-fat group, the central tendency and interquartile ranges are nearly identical. This suggests that fat category alone does not appear to lead to noticeable differences in how recipes are rated. 

Together, these visualizations imply that user ratings are generally high and that any difference between low-fat and high-fat recipes is subtle. A formal hypothesis test is therefore needed to determine whether the small observed differences are statistically significant.

### Interesting Aggregates

To further explore patterns in our dataset, we compute several aggregated statistics by grouping recipes based on different characteristics. Since our research question focuses on the relationship between fat content and recipe ratings, we examine how nutritional values and rating patterns differ across various recipe categories.

#### Average Rating by Fat Category

We start by grouping recipes into low-fat and high-fat categories (based on the mean `prop_fat`) and computing the average rating for each group. This aggregation helps us get an initial sense of whether high-fat or low-fat recipes tend to receive higher ratings on average.

| fat_category | value     |
|--------------|-----------|
| high-fat     | 4.641761  |
| low-fat      | 4.607899  |

#### Summary of Nutritional Values by Fat Category

Next, we compute summary statistics (mean and median) for several nutritional fields—such as calories, total fat, sugar, and saturated fat—within each fat category. This allows us to better understand how the nutritional profiles of high-fat and low-fat recipes differ.

| fat_category | calories_mean | calories_median | total_fat_mean | total_fat_median | sugar_mean | sugar_median | sodium_mean | sodium_median | protein_mean | protein_median | sat_fat_mean | sat_fat_median |
|--------------|---------------|-----------------|----------------|------------------|------------|--------------|-------------|----------------|---------------|----------------|---------------|-----------------|
| high-fat     | 494.766371    | 363.1           | 49.347361      | 34.0             | 54.345580  | 18.0         | 29.271498   | 17.0           | 37.376910     | 22.0           | 61.282781     | 40.0            |
| low-fat      | 360.742906    | 254.9           | 14.782967      | 9.0              | 83.942639  | 30.0         | 28.589766   | 12.0           | 28.605507     | 14.0           | 17.796043     | 8.0             |

#### Pivot Table: Fat Category × Rating Count

To explore how users rate recipes across categories, we create a pivot table summarizing the number of ratings each fat group receives. This helps us see whether one category is more frequently reviewed or has a different rating distribution pattern.

These aggregated statistics provide additional structural insight into the dataset and motivate further hypothesis testing.

| fat_category | avg_rating |
|--------------|------------|
| high-fat     | 41865      |
| low-fat      | 39308      |

The aggregated statistics provide several insights into how nutritional characteristics differ between low-fat and high-fat recipes, as well as how users rate them.

First, when comparing the mean average ratings, high-fat recipes (4.64) and low-fat recipes (4.61) receive very similar scores overall. Although the difference is small, high-fat recipes appear to have a slightly higher average rating. This difference, however, is small enough that we cannot draw conclusions without formal hypothesis testing.

The nutritional summary clearly reflects meaningful contrasts between the two groups. High-fat recipes have substantially higher levels of calories (mean 494.8 vs. 360.7), total fat (mean 49.35 vs. 14.78), and saturated fat (mean 61.28 vs. 17.80). In contrast, low-fat recipes have much higher sugar levels (mean 83.94 vs. 54.35), which indicates that low-fat foods in this dataset may compensate with additional sweetness. Protein levels are also moderately higher in high-fat recipes.

The pivot table summarizing rating counts shows that high-fat recipes (41,865 ratings) and low-fat recipes (39,308 ratings) receive a fairly comparable number of ratings. This balance indicates that any difference we observe in average ratings is unlikely to be driven by one group having substantially more user feedback than the other.

Overall, these aggregated patterns suggest that while nutritional profiles differ greatly between fat categories, their average ratings remain surprisingly similar. This motivates the need for a formal hypothesis test to determine whether the small observed difference in ratings is statistically significant.

## Assessment of Missingness

To learn the missingness of the aggregated dataset, we first look at the missing values in the dataset. The result is as follows:

| Column Name     | Missing Count |
|-----------------|----------------|
| name            | 1              |
| id              | 0              |
| minutes         | 0              |
| contributor_id  | 0              |
| submitted       | 0              |
| tags            | 0              |
| nutrition       | 0              |
| n_steps         | 0              |
| steps           | 0              |
| description     | 114            |
| ingredients     | 0              |
| n_ingredients   | 0              |
| calories        | 0              |
| total_fat_PDV   | 0              |
| sugar_PDV       | 0              |
| sodium_PDV      | 0              |
| protein_PDV     | 0              |
| sat_fat_PDV     | 0              |
| carb_PDV        | 0              |
| prop_fat        | 0              |
| fat_category    | 0              |
| user_id         | 1              |
| recipe_id       | 1              |
| date            | 1              |
| rating          | 15036          |
| review          | 58             |

Three columns, `description`, `rating`, and `review`, in the merged dataset have a significant amount of missing values, so we decided to assess the missingness on the dataframe.

### NMAR Analysis

The `description` column contains text written by recipe contributors, and writing a description is entirely optional on Food.com. Because contributors can freely choose whether or not to include a description, the missingness of this column is tied to the *unwritten content itself* rather than to other observable variables. This makes `description` a strong candidate for NMAR (Not Missing At Random).

A recipe may lack a description for reasons such as:
- the contributor felt the recipe was too simple to require explanation,
- they did not wish to spend extra time writing one,
- they intentionally left it blank because they believed the title or ingredients were sufficient,
- or the contributor did not have additional commentary to provide.

In all of these cases, the probability that `description` is missing depends directly on the **true (unobserved) description value**—that is, whether the contributor intended to provide text or not. This dependence cannot be fully explained or predicted by any of the observed variables in the dataset (such as `minutes`, `n_steps`, or nutritional values).

For this reason, the missingness mechanism for `description` is best characterized as **NMAR**, since the absence of a description is determined by unobservable internal factors (contributor intention) rather than random chance or relationships with other observed columns.

### Missingness Dependency

After identifying `rating` as a non-trivially missing column in the merged dataset, we examined whether its missingness depends on other columns. Specifically, we tested the dependency of missingness in `rating` on two variables: **`fat_category`** (whether a recipe is low-fat or high-fat) and **`minutes`** (the cooking time of the recipe). These tests help us understand whether the missingness mechanism for `rating` is systematic or random.

#### Fat Category and Rating Missingness

**Null Hypothesis:**  
The missingness of `rating` does *not* depend on the fat category of the recipe.

**Alternative Hypothesis:**  
The missingness of `rating` *does* depend on the fat category of the recipe.

**Test Statistic:**  
The absolute difference in the proportion of missing ratings between the high-fat group and the low-fat group.

**Significance Level:** 0.05

We generated 1000 permuted datasets by randomly shuffling the missingness indicator of `rating` to create a null distribution for the test statistic.

**Results**

<iframe
  src="asset/Missing_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed difference of 0.0037 is indicated by the red vertical line in the empirical distribution plot. Since the observed p-value（0.0） is **<< 0.05**, we reject the null hypothesis.  
**Conclusion:** The missingness of `rating` depends on `fat_category`. High-fat and low-fat recipes do not share the same missingness rate.

#### Minutes and Rating Missingness

**Null Hypothesis:**  
The missingness of `rating` does *not* depend on the cooking time (`minutes`) of the recipe.

**Alternative Hypothesis:**  
The missingness of `rating` *does* depend on the cooking time of the recipe.

**Test Statistic:**  
The absolute difference in the mean cooking time between the group with missing ratings and the group without missing ratings.

**Significance Level:** 0.05


We again performed a permutation test by shuffling the missingness indicator of `rating` for 1000 iterations, computing the test statistic each time to form a null distribution.

**Results**

<iframe
  src="asset/Missing_distribution2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed test statistic of 51.45 appears as the red vertical line. The resulting p-value（0.118） is **greater than 0.05**, so we fail to reject the null hypothesis.  
**Conclusion:** The missingness of `rating` does not depend on the cooking time (`minutes`) of a recipe.

## Hypothesis Testing


To investigate whether users rate low-fat and high-fat recipes differently, we conduct a hypothesis test comparing the average ratings of the two groups. This test is independent of any missingness analysis and focuses solely on determining whether fat content is associated with meaningful differences in user ratings.

### Hypotheses

**Null Hypothesis (H₀):**  
The average rating of low-fat recipes is the same as the average rating of high-fat recipes.  
In symbolic form:  
$$
H_0: \mu_{\text{low-fat}} = \mu_{\text{high-fat}} \\
$$

**Alternative Hypothesis (H₁):**  
The average rating of low-fat recipes is different from that of high-fat recipes.  
This is a two-sided test:  
$$
H_1: \mu_{\text{low-fat}} \ne \mu_{\text{high-fat}}
$$

This hypothesis is motivated by the question of whether users prefer richer, higher-fat recipes or leaner, healthier ones. Observing a meaningful difference would suggest that nutritional factors influence how users evaluate recipes.

### Test Statistic

We use the **absolute difference in mean rating** between the low-fat and high-fat groups as the test statistic:

$$
T = |\bar{r}_{\text{low-fat}} - \bar{r}_{\text{high-fat}}|
$$

Using the absolute difference ensures a two-sided test and measures the magnitude of rating disparity between the two fat categories, independent of direction.

In the permutation test, we will repeatedly shuffle the `fat_category` labels while preserving the distribution of ratings, compute the test statistic for each shuffled dataset, and compare these values to the observed test statistic. The resulting empirical p-value will allow us to decide whether to reject the null hypothesis.


### Procedure

We perform a permutation test by:

1. Computing the observed test statistic using the original data.  
2. Randomly permuting the `fat_category` labels 1,000 times.  
3. For each permutation, recomputing the test statistic.  
4. Comparing the observed statistic to the distribution of permuted statistics.  
5. Calculating the empirical p-value.

<iframe
  src="asset/Hypo_diagram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot above shows the null distribution of our test statistic — the difference in mean ratings between the two groups (low-fat and high-fat recipes) under the assumption that fat category has no effect on rating. This distribution was created by repeatedly shuffling the fat category labels and recomputing the mean difference for 1000 permutations.

The observed test statistic, represented by the red vertical line, is approximately **0.034**, which lies far to the right of the null distribution. None of the permuted differences come close to this value, indicating that such a large difference is extremely unlikely to occur by random chance alone.

Because the observed statistic falls well outside the range of the null distribution, the resulting p-value is **very close to 0**, which is far below the typical significance level of 0.05.

**Conclusion:**  
We reject the null hypothesis. There is statistically significant evidence that the average rating differs between high-fat and low-fat recipes. In other words, recipe ratings appear to be associated with fat category.

## Framing a prediction problem


In this section, we identify and clearly define a prediction problem based on the Food.com recipes and interactions data. Our goal is to leverage the cleaned dataset to build a predictive model that estimates how users will rate a recipe.

### Prediction Problem

**Prediction Task:**  
Predict the rating (1–5) that a user will give to a recipe based on the recipe’s nutritional profile and metadata.

**Response Variable:**  
`avg_rating` — the average numerical rating (1–5) given by a user.

**Type of Problem:**  
This is a **multiclass classification** problem because the response variable takes on one of five discrete values (1, 2, 3, 4, 5).

### Justification for Choosing the Response Variable

The rating a recipe receives is one of the most meaningful behavioral outcomes in this dataset, and predicting ratings is aligned with the broader theme of understanding user preferences. It is also practically useful: a good predictive model could support recipe recommendation systems or help recipe authors optimize their content.

### Features Used for Prediction

To ensure that only information available is included, we restrict our predictors to recipe-level attributes that exist before any user interacts with the recipe. These include:

- **Nutritional Features:**  
  - `calories`, `total_fat_PDV`, `sugar_PDV`, `sodium_PDV`,  
    `protein_PDV`, `sat_fat_PDV`, `carb_PDV`
- **Derived Features:**  
  - `prop_fat` (proportion of total fat PVD out of calories)
  - `fat_category` encoded as a boolean variable  
    (`1` = high-fat, `0` = low-fat)
- **Other Recipe Attributes:**  
  - `minutes` (time required to prepare the recipe)  
  - `n_steps` (number of steps)  
  - `n_ingredients` (number of ingredients)  
  - `tags` (processed into features if needed)


We explicitly exclude `avg_rating` or any interaction-level variables such as `review` because these occur *after* the rating is produced and would violate temporal causality.

### Evaluation Metric

We will evaluate our classifier using **F1-score** rather than accuracy.
 
F1-score accounts for both precision and recall and is therefore more informative in the presence of class imbalance. It allows us to measure how well the model performs across all rating levels rather than favoring the majority class.

## Baseline model


For the baseline model, I use a simple **multiclass classification** model to predict the rating (1–5).  
The baseline uses only a small set of easy-to-compute features available before a user leaves a review:

- `minutes`
- `n_steps`
- `n_ingredients`
- `fat_category` (encoded as a boolean: 1 for high-fat, 0 for low-fat)

For the baseline classifier, I chose **Logistic Regression (multinomial)** because it is simple, fast, and provides a reasonable reference point for more complex models. The evaluation metric is **accuracy**, since the dataset is relatively balanced across several rating categories and accuracy provides a straightforward baseline comparison.

My plan for improving the model in Step 7 includes:
- adding additional nutritional features (e.g., `calories`, `total_fat_PDV`, `sugar_PDV`)
- adding text-derived features such as the length of the recipe name or description
- trying more flexible models such as Random Forest or Gradient Boosting

These improvements will help capture richer patterns in the data and potentially provide better predictive performance than the simple baseline.

### Classification report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **1** | 0.00 | 0.00 | 0.00 | 118 |
| **2** | 0.00 | 0.00 | 0.00 | 155 |
| **3** | 0.00 | 0.00 | 0.00 | 552 |
| **4** | 0.14 | 0.00 | 0.00 | 4185 |
| **5** | 0.69 | 1.00 | 0.82 | 11225 |

#### Overall Metrics

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.69 |
| **Macro Avg Precision** | 0.17 |
| **Macro Avg Recall** | 0.20 |
| **Macro Avg F1** | 0.16 |
| **Weighted Avg Precision** | 0.51 |
| **Weighted Avg Recall** | 0.69 |
| **Weighted Avg F1** | 0.57 |

This report shows that the baseline model has ill prediction, especially when it is rated between 1-4. And the overall perfornamce is weak as well. This is because the dataset is largely biased, most of the ratings is 5, so the training samples are not enough. Meanwhile, the features need to be rechose, and feature engineering is also required to improve the performance.

## Final model

For the final model, we selected a set of engineered and domain-motivated features, including `minutes`, `n_steps`, `n_ingredients`, `prop_fat`, `calories`, `fat_category`, and two engineered features: `log_minutes` and `steps_per_ingredient`. These variables were chosen because our exploratory analysis showed meaningful associations with recipe rating. For example, both `minutes` and `n_steps` help capture recipe complexity, which may influence user satisfaction. The variable `fat_category` reflects a nutritional dimension that our earlier missingness and exploratory analysis showed to correlate with rating patterns. Additionally, `log_minutes` reduces the skew caused by extremely long cooking times, while `steps_per_ingredient` captures recipe density, which may reflect how effortful or structured a recipe is. Together, these features aim to provide richer signals than those used in the baseline model.

To address the severe class imbalance in the rating distribution, we incorporated **SMOTE** into our modeling pipeline to synthetically oversample the minority rating classes. We chose *XGBoost** as the final modeling algorithm because gradient boosting methods often perform well on tabular data and can capture nonlinear interactions between features. Using **GridSearchCV**, we tuned several key hyperparameters, including `max_depth`, `n_estimators`, `learning_rate`, and `subsample`, which help control model complexity and reduce the risk of overfitting. The best combination identified was a `max_depth` of **6**, `n_estimators` of **200**, `learning_rate` of **0.05**, and `subsample` of **0.8**.

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.00      | 0.01   | 0.00     | 122     |
| 1     | 0.00      | 0.01   | 0.00     | 160     |
| 2     | 0.07      | 0.02   | 0.03     | 788     |
| 3     | 0.37      | 0.27   | 0.31     | 5608    |
| 4     | 0.60      | 0.71   | 0.65     | 9557    |
| **accuracy** | — | — | 0.51 | 16235 |
| **macro avg** | 0.21 | 0.20 | 0.20 | 16235 |
| **weighted avg** | 0.48 | 0.51 | 0.49 | 16235 |

The `macro F1-score` of the final model is **0.21**, representing a modest improvement over the baseline model. Compared to the baseline, the final model provides slightly better separation among minority classes. For instance, the F1-scores for `classes 0–3` increased from near zero to small but nonzero values, and the performance for the dominant class (rating 4 in your relabeled dataset) remained relatively strong. However, performance across classes remains highly uneven, and the model still struggles significantly with minority classes because their feature distributions overlap heavily with those of the majority class. The limited predictive improvement suggests that, despite SMOTE and feature engineering, the available features do not contain strong enough signals to differentiate low-rated recipes. Additionally, the inherent imbalance of the dataset and the subjective nature of user ratings likely constrain the model’s achievable accuracy.

Overall, while the final model improves slightly over the baseline, the modest gains indicate that recipe rating prediction remains a challenging task given the current feature space, and further progress would likely require richer behavioral, textual, or image-based features beyond those provided in the structured dataset.

## Fairness Analysis

### Choice of Groups (Defining X and Y)
For fairness analysis, I evaluate whether my final XGBoost + SMOTE classifier performs equally well across different types of recipes.
Since nutritional composition may influence user ratings differently, I choose groups based on fat_category, which categorizes recipes as:
Group X: low-fat recipes
Group Y: high-fat recipes
This grouping is meaningful because low-fat and high-fat recipes often attract different types of users, potentially producing different rating behaviors and, consequently, uneven predictive performance.

### Evaluation Metric
Since my task is multiclass classification, I use macro F1-score, which equally weights all rating categories and is sensitive to minority classes.
Macro F1 is appropriate because fairness concerns usually arise when a model performs well overall but poorly on specific subgroups.

### Hypotheses
Null Hypothesis (H₀)
The model is fair.
There is no meaningful difference in macro F1-score between low-fat and high-fat recipes; any observed difference is due to randomness.
Alternative Hypothesis (H₁)
The model is unfair.
The macro F1-score for low-fat recipes is lower than that for high-fat recipes.
This is a one-sided test because we are specifically worried about worse performance for Group X.

### Test Statistic
$$
T = F1（{\text{low-fat}}） - F1（{\text{high-fat}}） \\
$$
​	
 
If T is strongly negative, performance for low-fat recipes is worse.

### Conclusion

To conduct the fairness permutation test, we created a binary column `is_high_fat` to separate recipes into high-fat and low-fat groups based on their fat proportion. Using our final XGBoost + SMOTE model, we computed the macro F1-score for each group. The observed F1-scores were:

High-fat recipes: 0.189

Low-fat recipes: 0.208

This resulted in an observed test statistic (high - low) of **-0.0188**, suggesting initially that the model performs slightly worse on high-fat recipes.

We then ran a permutation test with 1000 permutations, shuffling the `is_high_fat` labels while keeping the model predictions fixed. This produced a null distribution of simulated differences under the assumption that the model is fair. The resulting **p-value was 0.0**.

Because the p-value is far below 0.05, we reject the null hypothesis that the model is fair. This means that the difference in F1-scores is unlikely to be due to random chance. Our final model appears to perform **significantly worse on high-fat recipes compared to low-fat ones**.









































