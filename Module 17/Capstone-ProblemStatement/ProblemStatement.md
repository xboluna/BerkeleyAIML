### Research Question

General binary image classification to determine if images are NSFW or not-NSFW (whether images are appropriate for general audiences/children).

### Data Source

The data source should be available through a download of the Firebase image set. There are also online sources on Kaggle if this imageset is unavailable.

### Techniques

The best classifier will likely be a Convolution Neural Network or another transformer. Following CRISP-DM protocol, it will be necessary to compare different classifiers, including pre-trained models such as BERT which can be fine-tuned to map classifications to a single binary classifier.

Other, more intepretable classification techniques may also work and should be tested.

### Expected Results

Ideally, the classifier performs well in reducing the number of False Negatives. False Positives should be avoided if possible, however the primary goal of the classifier is to enforce segregation of NSFW content from open spaces.

### Importance

This project has a direct business application, reduces necessary man-hours and could even be an improvement on accuracy compared to manual image flagging.
