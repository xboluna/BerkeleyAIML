# PracApp3: Comparing Classifiers

In this practical application, I compare the performance of 6 classifiers (Logistic Regression, Decision Tree, SVC-linear kernel, SVC-rbf kernel, K Neighbors and SGD Classifier) on a dataset for marketing bank products over the telephone. 

Dataset [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. [Article](CRISP-DM-BANK.pdf) for more info on data and features.

The primary goal of this application is to produce a model which performs well in predicting the characteristics of bank customers which are most likely to be converted. In this way, a bank ideally would target resources more efficiently to improve return-on-investment for man-hours telemarketing and reduce the sense of intrusion experienced by costumers who are unlikely to be interested in the product.

This analysis follows the CRISP-DM methodology by beginning first with a breakdown and understanding of the dataset, including preprocessing (mapping, one-hot encoding) of features. I begin by fitting a dummy classifier to define a baseline accuracy. I attempt feature reduction using elasticnet and PCA as part of data understanding, however the results are not strong enough to propagate further in the analysis. I then fit the dataset using all 6 aforementioned models and compare their performance. In my case, the Decision Tree Classifier performs best. I fine-tune the model to an accuracy which improves the baseline by 41\%. I assess the performance of the model using the Lift curve metric.

### Data Understanding

This data is of 17 Portuguese direct-marketing campaigns for bank deposition subscription. 

The data is split into two groups: bank client, campaign, outcomes and socioeconomic attributes.

Client:
- 1 - age (numeric)
- 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- 5 - default: has credit in default? (categorical: 'no','yes','unknown')
- 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
- 7 - loan: has personal loan? (categorical: 'no','yes','unknown')

Related with the last contact of the current campaign:
- 8 - contact: contact communication type (categorical: 'cellular','telephone')
- 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- 10 - day\_of\_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

Outcomes:
- 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- 14 - previous: number of contacts performed before this campaign and for this client (numeric)
- 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

Socioeconomic:
- 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
- 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
- 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
- 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
- 20 - nr.employed: number of employees - quarterly indicator (numeric)

### Dataset Preprocessing

The dataset has 41,118 entries and is not missing any values.

Let's analyze the individual columns.
1. `age, duration, campaign, ndays, previous, emp.var.rate, cons.price.idx, cons.conf.indx, euribor3m, nr.employed` are numeric sequential values.
2. `default, housing, loan` are labelled with strings but are binary.
3. `month, day_of_week` are labelled numeric sequential
4. `job, marital, education, contact, poutcome` are nonsequential nonbinary labels. They will need to be one-hot encoded.
5. `y` is in yes/no binary strings.

Let's map the values in 3. and 5. to the appropriate numeric value (map strings to appropriate number). We can then create a transformer for the values in 2. and 4. to be one-hot encoded, making special note to include `drop='if_binary'` to avoid autocorrelated features for columns in 2.

Lastly, let's apply a standard scaling to avoid biasing high-value columns. This dataset is then ready to be split into training and test datasets.

### Baseline performance

Let's first perform a dummy classification to determine the baseline performance of a random classifier.

Clearly, there are far more 'no' than 'yes' -- 88.7\% of customers are not converted.

![](Images/DummyClassifier.png)

### Feature Selection

This section has become an aside; I attempted to perform some feature selection to decrease the computation time of certain models -- such as the rbf and poly SVCs.

I did this first with the SGDClassifier. I chose to use elasticnet (L1 + L2) to impose a balanced regularization on the features. I sampled a loss surface with the regularization strength (alpha) and picked the best-performing classifier. This resulted in just 15 non-zero features. When applying this reduction to the dataset, however, downline model performance was affected to an order of roughly 10%.

![](FeatureSelection.png)

Next, I attempted to reduce the dimensionality of the dataset by retaining just 90% of the feature variance. Again, this reduction affected model performance poorly.

![](Images/PCA.png)

As a result, I decided simply to run the subsequent for subsequently smaller sampling ranges, manually.

### Comparing Models

I compared the performance of 6 models: Logistic Regression, Decision Tree, SVC-linear kernel, SVC-rbf kernel, K Neighbors and SGD Classifier. I also tested SVC-polynomial kernel, however its performance was no better than other classifiers, and its fit-time was significantly poorer.

My selection process for the parameter space was to first sweep a area with low-samples (i.e. 10 fits across 10 orders of magnitudes) and to slowly converge towards the best-performing range and producing a fit over a range of roughly 100 samples. This step was taken because the fit time of certain models (especially SVC) was far too long. As discussed above, feature reduction negatively impacted the performance of some models. As a result, this approach was chosen instead.

![](Images/ModelPerformance.png)

Evidently, the Logistic Regression and Decision Tree perform better than most other models. Let's take the Decision Tree and continue fine-tuning the model.

### Fine-tuning the Decision Tree

![](Images/DTreeFineTuning)

The resulting training and validation scores are 0.920 and 0.914, respectively. These correspond to improvements of 40.7 and 30.9\% over the dummy classifer.

With the improvement on the dummy classifier exceeding 40% and 30% for the training and validation sets respectively, we can be confident that there exists tangible improvement to accuracy for customer conversion prediction. Our next step is to visualize the improvement in efficiency which the model could produce as a function of the population decile. This is the lift curve, which essentially quantifies the percentage improvement over the baseline of the modelling approach. As we can see, this model performs better than its raw accuracy might imply in predicting the rate of positive conversions(gain).

![](Images/LiftChart.png)

Lastly, we perform feature permutation to inspect the individual features importance to the model. We find that duration, nr.employed, month and euribor3m have the strongest impact on model decision-making. The first feature makes intuitive sense -- customers that stay on the line longer are more likely to be interested in purchasing the product. The economic indicators (nr.employed and euribor3m) are interesting as they have associated seasonality trends (i.e. month) which could indicate to a business when to spin up or down operations.

![](Images/FeatureImportance.png)

### Findings

This model can safely be employed with the expectation that the phone banking conversion rate will be improved by a factor of > 30%. This represents a concomitant efficiency improvement in worker hours ROI. The expanded dataset used to fit this model numbers 49 features --  a featurespace which is difficult to disentangle without this model. Thanks to the model, we now have the benefit of knowing that larger economic trends and seasonality have an effect on the efficiency of conversion.

Further improvements to this model could be made by comparing models of feature regularization as shown in the data understanding section. This would directly benefit the SVM fit time as shown in the average fit time in the modelling section. My intuition, drawing from the results of the attached paper, is that the SVM likely would have produced better results if its hyperparameters were better tuned. This was difficult to do at scale, however. 
