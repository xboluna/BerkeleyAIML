Drawing from your day-to-day experience or research, please provide a use case for implementing the CRISP-DM framework: business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

Discuss some of the strengths and weaknesses of applying it to the use case you provided. Additionally, please discuss any frameworks (i.e., TDSP or OSEMN) that you have applied that may be useful to this discussion.

---

My research involves looking for particular characteristics in a dataset of astronomical sources. In order to test our models, we have to find sources which match the criterion for good candidates. 

In this case, business understanding comes from interpreting what we are looking for: namely, the physics in our model may not match the generalized criteria which we allow exactly, HOWEVER the set of 'good candidates' must differ in a consistent way from what we expect.

As such, when we parse through data and enforce criteria, we must intentionally widen parameters and explore correlations carefully to determine if there are patterns which offset data systematically. We also need to generate new data on the existing dataset which will help inform conclusions, such as the hardness of the energy spectrum, etc. 

In preparing the data, we add additional parameters could reveal covariates which we weren't initially expecting. We also want to clean the dataset of anomalous data -- such as infinite or null values -- carefully inspecting it before we drop it in case the results are mistakes made my the creators of the dataset.

Finally we can explore how our models compare to the different levels of criterion which we enforced on the dataset. We evaluate the dataset on how well the data fits the model. We watch both the basic correlation AND if there exist any systematic differences from the model (including not just constant, but perhaps even poly differences) to see if the model fits well OR if it could be reasonably adapted to fit the data better.

From there, we can collate our results and compare them to our initial hypothesis for the project. Taking the data as a guide, we amend our understanding of the physics and make guesses about what contributes to the difference in our original model and the data.

In this use-case, the CRISP-DM process is applicable for managing the scope of the project (as many science projects often balloon). Additionally, the process encourages understanding of the data and model which allows wiggle-room to modify the model appropriately and explore other possibilities.
