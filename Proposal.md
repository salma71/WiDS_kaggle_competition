# DATA698 Project Propsal

*Description of the problem*

The ability to obtain a rapid understanding of the context of a patient’s overall health can be crucial to medical outcomes. Patients go to hospitals for a variety of reasons. These reasons range from benign to severe and the state of patients can also range from alert to unconscious. Gathering information that is important for treatment requires processing of patient data but the lack of verified medical histories and the structural inefficiencies of obtaining medical records that take days to transfer pose a problem for medical staff who would benefit from quickly knowing the presence of certain chronic conditions such as heart disease, injuries, or diabetes in a patient. Additionally, patients can also be uncooperative or untruthful and withhold information. For these reasons, the ability to detect chronic condition from basic data can be of great value to make informed clinical decisions. 

Diabetes Mellitus, or simply diabetes is one of these chronic condition that is important for medical practitioners to be aware of when treating patients. Detecting the presence of diabetes from patient data is a supervised machine learning problem of binary classification and a number of tools such as discriminant analysis, logistic regression and naive Bayes can be utilized for detection of diabetes. Analysis of the dataset (specify dataset) can help discover insights useful for model building, inference and prediction. 

*Why it’s interesting*

This is an interesting problem because it is not limited to diabetes. Any number of chronic health conditions can be detected using the same methodology. It is also useful because it may be used to detect conditions that patients may not even know they had. This is a step in the direction of provided better patient specific healthcare. 

*What other approaches have been tried*

[Sparse Modeling Reveals miRNA Signatures for Diagnostics of Inflammatory Bowel Disease](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140155) - This paper evaluates whether miRNA expression profiling in conjunction with machine learning classification techniques is a suitable non-invasive test to diagnose inflammatory bowel disease (IBD), in particular Crohn's disease (CD) and ulcerative colitis (UC). The ML methods evaluated are based are penalized SVM models, namely LASSO SVM, elastic net SVM, SCAD SVM and elastic SCAD SVM. To evaluate the validity of the feature selection employed by the penalized SVMs, two ensemble random forests models were built for each classification problem. 

[Prediction of Diabetes using Classification Algorithms](https://www.sciencedirect.com/science/article/pii/S1877050918308548#:~:text=Diabetes%20is%20considered%20as%20one,an%20increase%20in%20blood%20sugar.&text=Therefore%20three%20machine%20learning%20classification,diabetes%20at%20an%20early%20stage) - This paper uses three machine learning classification algorithms namely Decision Tree, SVM and Naive Bayes to detect diabetes at an early stage using data from the Pima Indians Diabetes Database (PIDD) sourced from the UCI machine learning repository

[Classification and prediction of diabetes disease using machine learning paradigm](https://link.springer.com/article/10.1007/s13755-019-0095-z) - This paper uses Logistic Regression to identify the risk factors for diabetes and four classifiers, namely Naïve Bayes, Decision Trees, Adaboost, and Random Forest to predict the occurence of diabetes in patients. 

For referece, diabetes is normally diagnosed using either:

- Oral Glucose Tolerance Test OGTT: measures your body's response to sugar (glucose). This requires an overnight fast, a post fast blood sugar reading as a baseline followed by the ingestion of a glucose solution and suplementary blood sugar readings. A normal blood glucose level is lower than 140 mg/dL (7.8 mmol/L). A blood glucose level between 140 and 199 mg/dL (7.8 and 11 mmol/L) is considered impaired glucose tolerance, or prediabetes. A blood glucose level of 200 mg/dL (11.1 mmol/L) or higher may indicate diabetes.

- A1C Test: also called the glycated hemoglobin, glycosylated hemoglobin, hemoglobin A1C or HbA1c test. An A1C test result reflects the average blood sugar level for the past two to three months. The A1C test measures the percentage of hemoglobin proteins in the blood are coated with sugar (glycated). The higher the A1C level is, the poorer a patient's blood sugar control is and the higher the risk of diabetes complications. A test value below 5.7% is normal. 5.7% to 6.4% is diagnosed as prediabetes. Readings of 6.5% or higher on two separate tests indicates diabetes.

*Hypothesis*

The presence of Diabetes Milletus can be detected from patient data collected within the first 24 hrs in an Intensive Care Unit. A parsimonious binary classification model can be built using variable selection and regularization techniques. The classification threshold can be adjusted to maximize sensitivity (true positive rate) of detection. 

*How the solution will improve the problem*

The proposed solution solves the problem by providing medical staff with another tool for quick diagnosis without the need to potentially costly and slow tests. 
