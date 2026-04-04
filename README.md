# wine-recommendation-system

Content-Based RecSys folder includes:
- Data Preprocessing 
- Item-to-Item RecSys, based on both TF-IDF and BoW
- User-to-Item RecSys, based on TF-IDF, additionally with NER and lemmatization
- User-to-Item RecSys, based on BoW
- User-to-Item RecSys, based on BERT

In all of the notebooks we used a subset of the original dataset due to the high dimensionality of the data which often caused extremely long runtimes (3h+) and crashes. The metrics of the models above are all analyzed in a separate file inside of the content_based_recsys folder. For the evaluation, due to the high number of near duplicates, we did both a strict and soft evaluation.