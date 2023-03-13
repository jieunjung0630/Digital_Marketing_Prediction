import streamlit as st
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_breast_cancer # dataset for the use of another experiment 
import numpy as np
import pandas as pd
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
df = data = pd.read_csv('data/income_df.csv')
df = df[[    'Age',
             'Income',
             'MntFishProducts',
             'MntFruits',
             'MntGoldProds',
             'MntMeatProducts',
             'MntSweetProducts',
             'MntWines',
             'Mnt_Product',
             'NumCatalogPurchases',
             'NumDealsPurchases',
             'NumPurchases',
             'NumStorePurchases',
             'NumWebPurchases',
             'NumWebVisitsMonth',
             'Sum_AcceptedCmp', 
           ]]


# Display the DataFrame in the Streamlit app
st.dataframe(df)

# Use Seaborn to plot a pair plot of the first 10 features
selected_cols = df.corr()[['Sum_AcceptedCmp']].sort_values(by='Sum_AcceptedCmp', ascending=False)[:20].T.columns.tolist()
sns.pairplot(df[df.columns[:5].tolist() + ['Sum_AcceptedCmp']], hue='Sum_AcceptedCmp')

# Display the plot in the Streamlit app
st.pyplot()

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X = df.drop('Sum_AcceptedCmp', axis=1).values

y = df['Sum_AcceptedCmp'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier on the training set
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)

# Display the accuracy in the Streamlit app
st.success(f'Accuracy: {accuracy:.2f}')

# Build the Streamlit app
st.sidebar.title("Campaign Acceptance Predictor")

# Create a slider for each column in the DataFrame
user_input = []
for col in df.columns[:-1]:
    slider_input = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(slider_input)

# Make a prediction based on the user's input
prediction = clf.predict([user_input])

# Display the prediction
st.markdown('### Number of Acceptance')

# Calculate the false positive rate and true positive rate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test = pd.DataFrame({'Sum_AcceptedCmp': y_test})
predictions = pd.DataFrame({'pred': predictions})

# predictions['pred_acceptance'] =np.where(predictions['pred'] > 0, 1,0 )
# y_test['true_acceptance'] =np.where(y_test['Sum_AcceptedCmp'] > 0, 1,0 )
classes = y_test.Sum_AcceptedCmp.unique()

cm = confusion_matrix(y_test, predictions, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=classes)
disp.plot()
plt.show()


# fpr, tpr, thresholds = roc_curve(y_test.true_acceptance.values, pred.pred_acceptance.values)

# Plot the ROC curve
# import matplotlib.pyplot as plt
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# Display the plot in the Streamlit app
st.pyplot()


