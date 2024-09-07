import base64
import io

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, r2_score, mean_squared_error , root_mean_squared_error

from imblearn.over_sampling import SMOTE



def file_to_df(file):
    name = file.name
    extension = name.split(".")[-1]


    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif extension == "xml":
        df = pd.read_xml(file)
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "json":
        df = pd.read_json(file)
    
    return df

def corr_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'correlation_matrix.png'), unsafe_allow_html=True)


def determine_task_type(df, target_column):
    """
    Determine if the task is Classification or Regression based on the target column.
    """
    unique_values = df[target_column].nunique()
    if df[target_column].dtype == 'object' or unique_values < 20:
        return 'Classification'
    else:
        return 'Regression'
 
def build_model(df, target_column, split_size, seed_number):

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    
    missing_values = df.isnull().sum().sum()
    duplicated_rows = df.duplicated().sum()


    if dropDuplicates and duplicated_rows > 0:
        df = df.drop_duplicates()

    if fillMissingVlaues and missing_values > 0 :
        imputer = SimpleImputer(strategy='mean')
        df[df.columns] = imputer.fit_transform(df[df.columns])


    X = df.drop(columns=[target_column])

    Y = df[target_column]


    
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    st.markdown('**1.2. Dataset dimension**')
    st.write('X (features)')
    st.info(X.shape)
    st.write('Y (target)')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    if missing_values > 0 and not fillMissingVlaues:
        st.warning(f'The dataset contains {missing_values} missing values. Consider checking the "Fill the missing values" option in Additional techniques.')
    if duplicated_rows > 0 and not dropDuplicates:
        st.warning(f'The dataset contains {duplicated_rows} duplicated rows. Consider checking the "Drop duplicated Rows" option in Additional techniques.')



    st.markdown('**1.4. Dataset Statistics**:')
    st.write(df.describe())

    st.markdown('**1.5. Correlation Matrix**:')
    corr_matrix(df)
    


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100, random_state=seed_number)
    
    inferred_task = determine_task_type(df,target_column=target_column)

    if polynomialFeatures:
        if df.isnull().sum().sum() > 0:
            st.warning("Polynomial Features cannot be applied due to missing values.")
        else:
            poly = PolynomialFeatures(degree=2)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)

    if dataAugmentation and task == 'Classification' and inferred_task == 'Classification':
        if df.isnull().sum().sum() > 0:
            st.warning("Data Augmentation cannot be applied due to missing values.")
        else:
            sm = SMOTE(random_state=seed_number)
            X_train, Y_train = sm.fit_resample(X_train, Y_train)





    if task == 'Classification':
        if algorithm == 'SVM':
            base_model = SVC(random_state=seed_number)
        elif algorithm == 'KNN':
            base_model = KNeighborsClassifier()
        elif algorithm == 'Random Forest':
            base_model = RandomForestClassifier(random_state=seed_number)
        elif algorithm == 'Decision Tree':
            base_model = DecisionTreeClassifier(random_state=seed_number)

        if ov == 'Compare One-Vs-Rest (OVR)':
            model = OneVsRestClassifier(base_model)
        elif ov == 'Compare One-Vs-One (OVO)':
            model = OneVsOneClassifier(base_model)
        else :
            model = base_model

    elif task == 'Regression':
        if algorithm == 'SVM':
            model = SVR()
        elif algorithm == 'KNN':
            model = KNeighborsRegressor()
        elif algorithm == 'Random Forest':
            model = RandomForestRegressor(random_state=seed_number)
        elif algorithm == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=seed_number)

    if algorithm in ['SVM', 'KNN']:
        if df.isnull().values.any() and not fillMissingVlaues:
            st.warning(f"The selected algorithm '{algorithm}' does not support missing values. Please enable the 'Fill the missing values' option to proceed.")
            return

    if inferred_task != task:
        st.warning(f"The selected task '{task}' does not match the inferred task type based on the target column. The dataset suggests this task should be '{inferred_task}'.")
        return
    else : 
        model.fit(X_train, Y_train)

        # Make predictions
        Y_pred = model.predict(X_test)

        # Display results
        if task == 'Classification':
            st.subheader('2. Classification Report:')
            st.text(classification_report(Y_test, Y_pred, digits=3))
        elif task == 'Regression':
            st.subheader('2. Regression Report:')
            st.write(f'R-squared: {r2_score(Y_test, Y_pred):.3f}')
            st.write(f'MSE: {mean_squared_error(Y_test, Y_pred):.3f}')
            st.write(f'RMSE: {root_mean_squared_error(Y_test, Y_pred):.3f}')


    # Return the trained model
        return model



    
    # Train the model
    
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href
st.set_page_config(page_title='ML App',
    layout='wide')

#---------------------------------#
st.write("""
# Machine Learning Prediction App""")


with st.sidebar.header('1. Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Please Upload the data", type=["csv","tsv","xlsx","json","xml"])

    

 

#  In the 1st application, a user will be able to upload a dataset with labels and have to choose if it is a classification or regression task, 
# then he will choose the target column and then you have to create and compare a set of ML models and show the results to him. You can give him
#  the option of selecting if he wants to also apply polynomial features (and choose the degree), decide if he wants to apply data augmentation
#  using SMOTE and also if he wants to include in the comparison One-Vs-Rest and One-Vs-One classifiers.

with st.sidebar.header('2. Task You Will Do: '):

    task = st.sidebar.radio(
    "2.1 Choose the type of task:",
    ('Classification', 'Regression'),
    help="Choose the type of task:\n- Classification: A type of supervised learning where the output variable is a label or category.\n- Regression: A type of supervised learning where the output variable is a continuous value."
)
with st.sidebar.header('3. About the algorithm: '):

    algorithm = st.sidebar.radio(
    "3.1 Choose the algorithm of task:",
    ('SVM', 'KNN', 'Random Forest', 'Decision Tree'),
    help="Choose the type of task:\n- SVM (Support Vector Machine): A classification algorithm that finds the hyperplane that best separates the classes in the feature space.\n- KNN (K-Nearest Neighbors): A classification (or regression) algorithm that assigns a label based on the majority label among the k-nearest neighbors to a data point.\n- Random Forest: An ensemble method that uses multiple decision trees to improve classification accuracy and control overfitting.\n- Decision Tree: A model that makes decisions based on answering a series of questions about feature values, leading to a prediction based on the majority class in the leaf node."
)

    split_size = st.sidebar.slider(
    '3.2 Data split ratio (% for Training Set)',
    min_value=10,
    max_value=90,
    value=75,  # Default value
    step=5,
    help="Specify the percentage of the data to be used for the training set. A typical split ratio is 75% for training and 25% for testing. Adjusting this ratio can impact the model’s performance."
)

    seed_number = st.sidebar.slider(
    '3.3 Set the random seed number',
    min_value=1,
    max_value=100,
    value=42,  # Default value
    step=1,
    help="Set the random seed number to ensure reproducibility of your results. Using the same seed number allows for consistent results across different runs."
)

with st.sidebar.header('4. Additional techniques: '):
    
    fillMissingVlaues = st.sidebar.checkbox('Fill the missing values')
    dropDuplicates = st.sidebar.checkbox('Drop duplicated Rows')


    polynomialFeatures = st.sidebar.checkbox('Polynomial Features')
    dataAugmentation = st.sidebar.checkbox('Data Augmentation')

    if task == 'Classification':
        ov = st.sidebar.radio('Comparaison Multi Class Classification',
    ('None' , 'Compare One-Vs-Rest (OVR)', 'Compare One-Vs-One (OVO)'),
)

        # ovr = st.sidebar.checkbox('Compare One-Vs-Rest (OVR)')
        # ovo = st.sidebar.checkbox('Compare One-Vs-One (OVO)')
    else:
        ov = False



st.subheader('1. Dataset')

if uploaded_file is not None:

    df = file_to_df(uploaded_file)
    # st.dataframe(df)

    # df = pd.read_csv(uploaded_file)


    st.markdown('**1.1. Glimpse of dataset**')
    
    show_all = st.checkbox('Show entire dataset')
    
    if show_all:
        st.write(df)
    else:
        st.write(df.head())

    target_column = st.selectbox(
        "1.1.1 Select the target column:",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,  
        help="Select the target column for the task. By default, the last column is selected."
    )







    build_model(df, target_column, split_size, seed_number)



else:
    st.info('Awaiting for the file to be uploaded.')
