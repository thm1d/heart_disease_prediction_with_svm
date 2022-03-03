
# Heart Disease Prediction With SVM

## Heart Disease Data
- Data from three hospital systems: Cleveland, Hungary, Switzerland 
- This database contains 76 attributes, but all published experiments refer to a subset of 14 of them 
- The goal field refers to the presence of heart disease in the patient, an integer valued from 0 (no presence) to 4 
- Names and social security numbers of patients were removed from the database, replaced with dummy values

## Heart Disease Dataset
- 13 Independent Variables
- Prediction Output (0-4)

![Dataset](https://i.ibb.co/vqcqNrm/Screenshot-196.png)

## Implementation 
- Missing value handled using `SimpleImputer`
- Splited test data and train data
- `StandardScaler`is used to fix class imbalance
- Support Vector Classification(SVC) is used and kernel used is rbf as others(linear, poly, sigmoid) give less accuracy


