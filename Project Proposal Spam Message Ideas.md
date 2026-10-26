# Comparing LSTM and Transformer Models for SMS Spam Classification

## 1. Task and Research Question (2 points)

- **Task:** Binary text sequence classification. Given a short SMS message, predict whether it is spam or a legitimate message (ham).
- **Research question:** How do an LSTM and a Transformer encoder compare when classifying SMS messages as spam or ham, and how much does handling the strong class imbalance change each model's performance?
- **Goal:** Compare the models using imbalance-aware classification metrics, examine how the choice of imbalance treatment affects precision and recall, and inspect examples where their predictions are incorrect.
    

## 2. Proposed Models (2 points)

- **Model 1:** LSTM classifier
	- Convert message text into tokens and learned word embeddings.
	- Process the variable-length token sequence with an LSTM layer. 
	- Pass the final hidden representation through a linear classification layer to predict spam or ham.
    
- **Model 2:** Transformer encoder classifier
	- Convert tokens into embeddings and add positional encodings.
	- Process the sequence using Transformer encoder layers with multi-head self-attention and feed-forward networks.
	- Pool the sequence representations with a masked mean and use a linear layer for binary classification.
### Training and Evaluation:   

- Implement both models in PyTorch.
- Use binary cross-entropy with logits, an Adam optimizer, and validation data for selecting model settings.
- Report accuracy, precision, recall, F1-score, PR-AUC, and confusion matrices, since accuracy alone is misleading when only about one message in eight is spam.
- Treat class-imbalance handling as a controlled condition: compare unweighted loss, class-weighted loss, and validation-set threshold tuning on the same splits.
- Analyze performance across short and long messages and on deliberately obfuscated spam (misspellings, inserted characters, phone numbers, and links).
- Begin with small experiments and verify that each model can overfit a single training example before full training, and repeat runs with fixed seeds.

## 3. Dataset and Data Exploration (4 points)

### Dataset: 
- [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset

### Size and labels:

- Approximately 5,600 labelled SMS messages, of which roughly 13% are spam and 87% are ham.
- Binary labels: spam and ham, with a strongly imbalanced class distribution.
- Messages are short, informal, and often contain abbreviations, links, phone numbers, and irregular spelling.
- The collection contains several hundred exact duplicate messages, which must be removed before splitting to prevent train/test leakage.
- Access: The dataset is publicly accessible. We will verify and document the dataset card's license and usage terms before finalizing the proposal, and cite the original collection paper.
- Planned exploration:
- Load the dataset and verify the number of examples and class distribution.
- Check for missing text, empty messages, and duplicate messages, and decide on a de-duplication rule before splitting.
- Calculate minimum, median, and maximum message lengths and compare length distributions between spam and ham.
- Inspect representative spam and ham messages, and plot the message-length distribution.
- Count links, phone numbers, digit and uppercase ratios, and vocabulary size to decide how much preprocessing is justified.
- Record observed findings and use them to select a reasonable maximum sequence length for model training.
- De-duplicate first, then create a stratified train/validation/test split. Reserve the test split for final evaluation, and fit the vocabulary and preprocessing settings using training data only to prevent data leakage.

## 4. Ethical Considerations (2 points)
- Spam filtering protects users from fraud, phishing, and unwanted advertising, and could be applied to similar short-message channels.
- Potential harms: Errors are asymmetric. A missed spam message can expose a user to a scam, while a false positive silently blocks a legitimate message such as a bank alert, a delivery notice, or an appointment reminder. We will report false-positive rates explicitly rather than optimizing accuracy alone.
- Differences across groups: Messages may be written in non-standard English, dialects, or other languages, and legitimate messages that contain links or promotional wording are more likely to be flagged. Uneven error rates across these categories would disadvantage some users.
- The dataset comes from a particular set of sources and time periods, and its spam definition reflects the collectors' judgment rather than a universal one. Legal definitions of spam also vary by region.
- Models may learn shortcuts, such as the presence of a number or a particular phrasing, that do not generalize. We will examine misclassified examples, acknowledge dataset limitations, and avoid treating model predictions as definitive judgments about a sender.

## 5. Division of Team Responsibilities (2 points)

Member 1 — Data preparation and baseline
- Download and document the dataset and its license.
- Inspect class balance, message lengths, missing values, and duplicates, and define the de-duplication rule.
- Implement tokenization, data loading, and stratified train/validation/test handling.
- Build a simple majority-class and bag-of-words baseline and produce dataset statistics and plots.

Member 2 — LSTM implementation
- Implement the embedding layer, LSTM, and binary classification head.
- Handle variable-length sequences and padding appropriately.
- Verify input and output dimensions and the loss calculation.
- Test overfitting on a single example, train the model, and record validation and test results.
    

Member 3 — Transformer implementation
- Implement token embeddings, positional encodings, Transformer encoder layers, masked pooling, and the classification head.
- Implement attention masks to handle padded tokens correctly.
- Verify the forward pass and overfit a single example.
- Train and evaluate the model and record validation and test results.

Member 4 — Experimental evaluation and error analysis
- Implement a shared evaluation pipeline for accuracy, precision, recall, F1-score, PR-AUC, and confusion matrices.
- Run the imbalance conditions and record how class weighting and threshold tuning change precision and recall for each model.
- Compare the LSTM and Transformer using the same test set and evaluation metrics.
- Analyze model performance across short and long messages and on obfuscated spam, and examine misclassified examples to identify potential reasons for incorrect predictions.
- Prepare comparison tables, visualizations, and the results section.

Shared responsibilities
- Agree on evaluation metrics, data splits, preprocessing methods, and experimental settings.
- Use validation data for model selection and threshold tuning, and reserve the test split for final evaluation.
- Review misclassified examples, ethical considerations, and dataset limitations together.
- Integrate and test the code, document random seeds and model settings, and reproduce the main results.
- Compare model performance, training time, and computational costs.
- Collaborate on the final report, presentation, and discussion of findings.

## Reference

Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11). UCI Machine Learning Repository, SMS Spam Collection Dataset.

## 1. Most relevant papers

Read first
**Deep learning and Transformer models**
1. "Optimizing SMS Spam Detection: Comparative Analysis of Hybrid Voting Ensembles and Bi-LSTM Networks"
Source: IEEE Xplore

The authors evaluate a custom Bi-LSTM model and compare it against several voting-ensemble methods on the SMS Spam Collection dataset, reporting 99.30% accuracy for the Bi-LSTM. Because this is the closest published setup to your project on the same dataset, it is the natural reference point for your LSTM results.

Why it matters to you: It shows that near-ceiling accuracy is already reported on this dataset using standard splits, so your contribution should be framed as an imbalance-aware comparison with error analysis, not as a search for higher headline accuracy.

[Read the paper](https://ieeexplore.ieee.org/abstract/document/10810777)

2. "Noise-Robust SMS Spam Classification Using a Hybrid BERT and Character-Level CNN Architecture"

Source: ScienceDirect

The authors propose a BERT and character-level CNN fusion model, reporting 99.28% accuracy and a 92.8% F1-score, with an explicit focus on robustness to noisy, obfuscated text.

Why it matters to you: It is the strongest evidence that character-level information helps with misspellings and inserted characters, which motivates the obfuscation slice of your error analysis and sets a realistic expectation for how your Transformer compares with a word-only model.

[Read the paper](https://www.sciencedirect.com/science/article/pii/S1877050926016959)

3. "Explainable AI for SMS spam filtering: A novel hybrid architecture combining fuzzy logic and bidirectional LSTM networks"

Source: ScienceDirect

This work integrates fuzzy logic with a bidirectional LSTM, reaching 99.4% accuracy on the UCI dataset while also producing explanations for its decisions.

Why it matters to you: It demonstrates that interpretability has already been attached to a recurrent spam classifier, so you should position your analysis of misclassified examples as a practical alternative to full explainability rather than as a novel contribution.

[Read the paper](https://www.sciencedirect.com/science/article/pii/S2773186325002518)

**Hybrid and ensemble methods**

4. "A Hybrid Ensemble SMS Spam Detection System through Weighted Soft Voting"

Source: IEEE Xplore

The authors combine Naïve Bayes, logistic regression, and random forest in a weighted soft-voting ensemble, reporting 97.85% accuracy on the UCI SMS Spam Collection.

Why it matters to you: It provides a strong non-neural baseline on exactly this dataset. Reporting it alongside your models shows whether the added complexity of a sequence model is justified on a dataset this small.

[Read the paper](https://xplorestaging.ieee.org/document/11318293)

5. "Innovative Spam Detection Using Hybrid Machine Learning Algorithms"

Source: Zenodo

The paper builds a hybrid of Naïve Bayes and a support vector machine using TF-IDF features together with non-textual features.

Why it matters to you: It shows that hand-crafted non-textual features such as digit or link counts carry real signal, which supports the feature statistics you plan to compute during data exploration and gives you a baseline to compare the learned representations against.

[Read the paper](https://zenodo.org/records/17657703#1)

  

**Planned experimental conditions**

|              |                                           |
| ------------ | ----------------------------------------- |
| Condition    | Setting                                   |
| 1. Baseline  | Unweighted loss, default threshold        |
| 2. Weighted  | Class-weighted binary cross-entropy       |
| 3. Threshold | Threshold tuned on validation predictions |
| 4. Combined  | Class weighting plus tuned threshold      |
| 5. Slices    | Best setting, short/long and obfuscated   |
|              |                                           |
