import streamlit
import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import plotly.express as px  # pip install plotly-express
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image  # Pillow  https://pillow.readthedocs.io/en/stable/reference/Image.html
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import pickle
from scipy import stats
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from scipy.stats import randint, uniform
from numpy import random
from dtreeviz.trees import dtreeviz
import plotly.graph_objs as go
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score



# This will display the title top left of your screen
st.set_page_config(
    page_title="Data Divers Team ",
    page_icon=":gem:",
    # layout="wide",     # "wide" uses the entire screen.
    layout="centered",  # Defaults to "centered", which constrains the elements into a centered column of fixed width
    initial_sidebar_state="expanded"
    # initial_sidebar_state="collapsed"
)


# Streamlit cache things which will help to run the application faster
@st.cache()
# NOTE: This must be the first command in your app, and must be set only once
# st.set_page_config(layout="wide")

def types(df):
    return pd.DataFrame(df.dtypes, columns=['Type'])

def lin_option():  # changed / is new for ED
    st.header('Transformed MLR Model')
    mod1_equation = r'''

        $
        \log_{10}{\hat {(Cost)}} = 
        \beta_{0} + 
        \beta_{1}(carat)^{\frac{1}{3}} + 
        \beta_{2}(cut) +
        \beta_{3}(clarity) +
        \beta_{4}(depth) +
        \beta_{5}(table) +
        \beta_{6}(year) +
        \beta_{7}(length) +
        \beta_{8}(width) +
        \beta_{9}(height) +
        \beta_{10}(colorD) +
        \beta_{11}(colorE) +
        \beta_{12}(colorF) +
        \beta_{13}(colorG) +
        \beta_{14}(colorH) +
        \beta_{15}(colorJ)
        $
        '''

    st.write(mod1_equation)
    st.markdown("""---""")
    st.header('Clarifications & Interpretations:')
    # clarification()
    st.markdown("""---""")
    st.header('Predictions:')
    # pred_permission = st.beta_expander('Predict Values')
    # return pred_permission


def clean(df, new=False):
    lin_df = df.copy()

    if new == True:
        lin_df.dropna(inplace=True)
    else:
        nan_carats = df[df['carat'].isnull()]
        nan_carats_index = nan_carats.loc[:, 'index']

        for index in nan_carats_index:
            lin_df.loc[lin_df['index'] == index, 'carat'] = lin_df[lin_df['index'] == index]['carat'].mean()

        temp = lin_df[lin_df['cost (dollars)'] > 0]
    # Rows with zero in any of the dimension columns
    zeros = lin_df[(lin_df['width (mm)'] == 0) | (lin_df['length (mm)'] == 0) | (lin_df['height (mm)'] == 0)]
    lin_df.drop(zeros.index, inplace=True)
    lin_df.drop(['index'], axis=1, inplace=True)
    return lin_df

def lin_reorg(df):  # cleaned NEW data for linear model ONLY

    # user-inputted csv is assigned to variable ### 'new_diamonds'
    try:
        lin_df = clean(df=df, new=True)
    except KeyError:
        st.error("Please input a newer file to clean!")
   # if lin_df == "['index'] not found in axis":
        #st.error('Choose another file to upload!')

    num_attributes = ['carat', 'depth', 'table', 'year', 'length (mm)', 'width (mm)', 'height (mm)']
    cat_attributes_ordinals = ['clarity', 'cut']
    cat_attributes_onehot = ['color']
    categories = [['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                  ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']]

    col_names = 'length (mm)', 'width (mm)', 'height (mm)'

    full_pipeline = ColumnTransformer([
        ('ordinal', OrdinalEncoder(), cat_attributes_ordinals),
        ('onehot', OneHotEncoder(), cat_attributes_onehot),
    ])

    lin_df[['clarity', 'cut', 'G', 'E', 'F', 'H', 'D', 'I', 'J']] = full_pipeline.fit_transform(lin_df)
    corrected_columns = ['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)', 'year', 'clarity',
                         'cut',
                         'D', 'E', 'F', 'G', 'H', 'J']

    columns = lin_df.columns
    lin_df = lin_df.loc[:, columns.isin(corrected_columns)]
    lin_df = lin_df[corrected_columns]
    return lin_df


def main():
    st.sidebar.title("What would you like to do?")
    # activities = ["Home","Exploring the data", "Plotting and Visualization","Prediction", "Our Team"]
    activities = ["Home", "Exploring the data", "Prediction", "Prediction using R", "Our Team"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # USING THE SIDE BAR
    st.sidebar.title("Please upload Your CSV File: ")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type={"csv"})
    og_data = pd.read_csv('https://raw.githubusercontent.com/KevinT-13/Hack.Diversity/main/cleaned_diamond.csv')
    # uploaded_file = st.file_uploader("Choose a CSV file")

    # if uploaded_file is not None and choice == "Home":
    if choice == "Home":
        # data = pd.read_csv(uploaded_file)

        # My title of my project
        st.title("We are Diamond Data Divers :gem:")

        IMAGE = "diamond.png"
        st.markdown(
            """
            <style>
            .container {
                display: flex;
            }
            .logo-text {
                font-weight:1000 !impotant;
                font-size:60px !important;
                color: #f9a01b !important;
                padding-top: 75px !important;
            }
            .logo-img {
                float:center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="container">
                <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(IMAGE, "rb").read()).decode()}">
                <p class="logo-text"></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
           \n![](../../../../Desktop/image_kevin.png.jpg)

            """,
            unsafe_allow_html=True,
        )
        st.subheader("About this project:")
        st.markdown('''
                  :large_blue_diamond: Data cleaning.\n
                  :large_blue_diamond: Understanding the diamond market.\n
                  :large_blue_diamond: Identify trends.\n
                  :large_blue_diamond: Build a model that can predict prices for 2022.\n
                   ''')

        # How to get rid of "Unnamed: 0" column in a pandas DataFrame read in from CSV file?
    # https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe-    read-in-from-csv-fil

    elif uploaded_file is not None and choice == "Exploring the data":

        # My title of my project
        st.title("Exploring the Data :chart_with_upwards_trend:")

        # data = pd.read_csv(uploaded_file, index_col=[0])
        data = pd.read_csv(uploaded_file)
        # st.header(choice)

        # Show dataset
        if st.checkbox("Show Dataset"):
            rows = st.number_input("Number of rows", 5, len(data))
            st.dataframe(data.head(rows))

        # Show columns
        if st.checkbox("Columns"):
            st.write(data.columns)

            # Show Shape
        if st.checkbox("Shape of Dataset"):
            data_dim = st.radio("Show by", ("Rows", "Columns", "Shape"))
            if data_dim == "Columns":
                st.text("Number of Columns: ")
                st.write(data.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows: ")
                st.write(data.shape[0])
            else:
                st.write(data.shape)

        # Show Data summary
        if st.checkbox("Show empty cells"):
            st.text("Datatypes Summary")
            st.write(data.isnull().sum())

        # Show a concise summary of the dataset
        # if st.checkbox("Getting a concise summary of the dataset"):
        # st.write(data.dtype())

        # Show Data summary
        if st.checkbox("Getting descriptive statistics of the data"):
            st.text("Datatypes Summary")
            st.write(data.describe())

        if st.checkbox("Viewing the counts of categorical data and the relationship between continuous variables"):
            type_of_plot = st.selectbox("Select Type of Plot",
                                        ["categorical data(Bar chart)", "continuous data(Scatter Plot)",
                                         "continuous data(Heatmap)", "continuous data(Histogram)"])
            if type_of_plot == "categorical data(Bar chart)":
                # Create a list of columns objects
                list_of_object_columns = []
                list_of_numerical_values_columns = []  # int and float

                for column_name, coltype in data.dtypes.iteritems():
                    if coltype == 'object':
                        list_of_object_columns.append(column_name)
                        categorical_data = list_of_object_columns
                    elif coltype == 'float64' or coltype == 'int64':
                        list_of_numerical_values_columns.append(column_name)
                        numerical_data = list_of_numerical_values_columns

                st.subheader("Bar Chart")
                column = st.selectbox("Pick an item : ", list_of_object_columns)
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x=column, data=data)
                st.pyplot(fig)
                # st.write("Bar Plot")
                # x_column1 = st.selectbox("Select a column for X Axis", categorical_data)
                # st.write(sns.catplot(x="color",data=data,kind="count", aspect=1.5))


            elif type_of_plot == "continuous data(Scatter Plot)":
                # Hide this warning
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # Create a list of columns objects
                list_of_object_columns = []
                list_of_numerical_values_columns = []  # int and float

                for column_name, coltype in data.dtypes.iteritems():
                    if coltype == 'object':
                        list_of_object_columns.append(column_name)
                        categorical_data = list_of_object_columns

                    elif coltype == 'float64' or coltype == 'int64':
                        list_of_numerical_values_columns.append(column_name)
                        numerical_data = list_of_numerical_values_columns

                scatter_x = st.selectbox("Select a column for X Axis", numerical_data)
                scatter_y = st.selectbox("Select a column for Y Axis", numerical_data)

                st.subheader("Scatter Plot")
                # st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = data))
                # st.pyplot()
                # st.title('Scatterplot between carat and price')

                fig = plt.figure()
                sns.scatterplot(x=data[scatter_x], y=data[scatter_y])
                st.pyplot(fig)

            elif type_of_plot == "continuous data(Heatmap)":
                st.subheader('Correlation: Heatmap')

                # WE CAN PLOT THIS WAY  OR
                # st.write(sns.heatmap(data.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))

                # WE CAN PLOT THIS WAY  OR
                # mask=np.triu(np.ones_like(data.corr()))
                # st.write(sns.heatmap(data.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}, mask=mask))
                # st.pyplot()

                # or THIS WAY!!!!!!!!!!!!!!!!!!!!!!!
                fig = plt.figure(figsize=(10, 5))
                st.write(sns.heatmap(data.corr(), annot=True, cmap='coolwarm'));
                st.pyplot(fig)


            elif type_of_plot == "continuous data(Histogram)":

                # Create a list of columns objects
                list_of_object_columns = []
                list_of_numerical_values_columns = []  # int and float

                for column_name, coltype in data.dtypes.iteritems():
                    if coltype == 'object':
                        list_of_object_columns.append(column_name)
                        categorical_data = list_of_object_columns

                    elif coltype == 'float64' or coltype == 'int64':
                        list_of_numerical_values_columns.append(column_name)
                        numerical_data = list_of_numerical_values_columns

                st.subheader("Histogram")

                # fig = plt.figure()

                ax = st.selectbox("Select a numerical item: ", numerical_data)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.write(sns.displot(data[ax], kde=False))
                st.write(plt.axvline(x=np.mean(data[ax]), color='red', label='mean'))
                st.write(plt.axvline(x=np.median(data[ax]), color='orange', label='median'))
                st.write(plt.legend(loc='upper right'))
                st.pyplot()

                # Show Shape
            # if st.checkbox("Histogram"):
            # data_dim = st.radio("Show by", ("All Histograms", "Choose Histogram"))
            # if data_dim == "Choose Histogram":
            # st.subheader("Histogram")
            # ax =st.selectbox("Select a numerical item: ", numerical_data)
            # st.write(sns.displot(data[ax],kde=False))
            # st.write(plt.axvline(x=np.mean(data[ax]), color='red', label='mean'))
            # st.write(plt.axvline(x=np.median(data[ax]), color='orange', label='median'))
            # st.write(plt.legend(loc='upper right'))
            # st.pyplot()
            # else:
            # fig = plt.figure()
            # data_dim == "All Histograms"
            # st.write(data.hist(bins=50, figsize = (20,15)))
            # st.pyplot(fig)

            # fig = plt.figure(figsize = (10,5))
            # st.write(sns.heatmap(data.corr(),annot = True , cmap = 'coolwarm' ));
            # st.pyplot(fig)

    elif uploaded_file is not None and choice == "Prediction":
        # st.subheader(choice)
        st.title("Data Modeling :chart_with_downwards_trend:")
        data = pd.read_csv(uploaded_file)
        df = data.copy()
        all_columns = df.columns.tolist()
        type_of_model = st.selectbox("Select Model Type", ["Linear Regression", "Tree"])  # changed this

        if type_of_model == "Linear Regression":  # new along with the things below this
            st.header('Transformed MLR Model')
            mod1_equation = r'''
                    $
                    \log_{10}{\hat {(Cost)}} = 
                    \beta_{0} + 
                    \beta_{1}(carat)^{\frac{1}{3}} + 
                    \beta_{2}(cut) +
                    \beta_{3}(clarity) +
                    \beta_{4}(depth) +
                    \beta_{5}(table) +
                    \beta_{6}(year) +
                    \beta_{7}(length) +
                    \beta_{8}(width) +
                    \beta_{9}(height) +
                    \beta_{10}(colorD) +
                    \beta_{11}(colorE) +
                    \beta_{12}(colorF) +
                    \beta_{13}(colorG) +
                    \beta_{14}(colorH) +
                    \beta_{15}(colorJ)
                    $
                    '''
            st.write(mod1_equation)
            st.write('RMSE: 1296.311')
            st.write('Adjust R^2: 95.00%')
            st.image('image_kevin.png.jpg')  # please insert image with blue lowess line here

            st.markdown("""---""")
            st.header('Clarifications & Interpretations:')
            st.write('What does the model mean:')
            st.write('''
                Simply put, the model is being used to predict log(cost)--a decision made based on failure to pass assumptions.
                Coefficients used are as explained in the above model (i.e., carat, cut clarity, etc.). Further below, the response 
                variable will be set to exp() in order to compare it to previous years' costs of wholesale diamonds. Lastly, this model will always be sensitive
                to outliers. 
                ''')
            st.write('Why not this model:')
            st.write('''
                    While the model has proven to be a good forecasting and predictive tool, its complexity and
                    transformations associated with the log(cost) and cube root of carat has removed a crucial element of any model:
                    interpretation of both the coefficients and key statistical values such as R-squared. Another key part as to 
                    why not this particular model is that the assumptions barely fit (normality, homogeneity, etc.) 
                    and as a result, the model was heavily influenced. In addition to the assumptions we can see that the linear model's RMSE
                    could be lowered by using an alternative machine learning method. 
                     ''')
            # clarification()
            st.markdown("""---""")
            st.header('Predictions:')
            st.subheader('Cleaned Input Dataset:')
            #if lin_reorg(df=data).empty == True:
                #st.error('cleaning did not work')
            lin_df = lin_reorg(df=data)
            st.write(lin_df.shape)
            st.write(lin_df.head(5))

            lin_df["cost (dollars)"] = (-36.70 + ((lin_df['carat'] ** (1 / 3)) * 3.935) + (
                    lin_df['depth'] * (-0.004282)) + (lin_df['table'] * (-0.01158)) +
                                        (lin_df['length (mm)'] * (-0.06204)) + (lin_df['width (mm)'] * 0.3773) +
                                        (lin_df['height (mm)'] * 0.007449) + (lin_df['year'] * 0.01971)
                                        + (lin_df['cut'] * 0.002227) + (lin_df['clarity'] * 0.06222)
                                        + (lin_df['D'] * 0.3447) + (lin_df['E'] * 0.3003) + (lin_df['F'] * 0.2949)
                                        + (lin_df['G'] * 0.2522) + (lin_df['H'] * 0.1221) + (lin_df['J'] * (-0.150)))

            lin_df['cost (dollars)'] = np.exp(lin_df['cost (dollars)'])
            st.subheader('Combined Dataset:')
            c_columns = ['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)', 'year', 'clarity',
                         'cut', 'D', 'E', 'F', 'G', 'H', 'J', 'cost (dollars)']
            c = og_data.columns
            og_data = og_data.loc[:, c.isin(c_columns)]
            og_data = og_data[c_columns]
            joined_df = lin_df.append(og_data, ignore_index=True)
            st.write(joined_df.shape)
            st.write(joined_df.head(5))
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # pd.set_option('display.float_format', lambda x: '%.4f' % x)
            #
            # joined_df['year'] = pd.to_numeric(joined_df['year'])
            #
            # joined_pivot = joined_df.pivot_table(index='year', values='cost (dollars)', aggfunc='sum', margins=True)
            # joined_plot = joined_pivot[:-1].plot()
            #plt.plot(joined_df.pivot_table(index='year', values='cost (dollars)', aggfunc='min', margins=True))
            # plt.ticklabel_format(axis="y", style="sci")
            # plt.title("Trend of Wholesale Diamond Prices")
            # plt.ylabel('Cost of Diamonds (in 100 Millions)')
            # plt.xlabel('Years')
            # st.write(joined_pivot)
            # st.pyplot()

            st.subheader('Total Cost of Diamonds per Year')
            st.image('dying.png')
            st.subheader('Trend of Wholesale Diamond Prices')
            st.image('deadM.png')

            st.markdown("""---""")


        elif type_of_model == "Tree":
            try:
                new_diamonds = lin_reorg(df=data)
            except KeyError:

                st.error("Please input a newer file to clean!")
            num_attributes = ['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)']
            cat_attributes_ordinals = ['clarity', 'cut']
            cat_attributes_onehot = ['color']
            categories = [['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                          ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']]

            col_names = 'length (mm)', 'width (mm)', 'height (mm)'

            length_ix, width_ix, height_ix = [new_diamonds[num_attributes].columns.get_loc(c) for c in col_names]

            class VolumeAdder(BaseEstimator, TransformerMixin):
                def __init__(self, add_volume=True):
                    self.add_volume = add_volume

                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    X = X.values
                    if self.add_volume:
                        volume = X[:, length_ix] * X[:, width_ix] * X[:, height_ix]
                        return np.c_[X, volume]
                    else:
                        return X

            @st.cache(persist=True)
            def split(og_data):
                y = og_data['cost (dollars)']
                replacement = og_data.drop(columns=["cost (dollars)"])
                x = replacement[['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)']]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

                return x_train, x_test, y_train, y_test
            x_train, x_test, y_train, y_test = split(og_data)

            if type_of_model == "Tree":
                st.subheader("Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                                       key="n_estimators")
                max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step=1, key="max_depth")
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"),
                                             key="bootstrap")

                if st.button("Classify", key="classify"):
                    st.subheader("Random Forest Results")
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                                   n_jobs=-1)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    x_valid = new_diamonds[['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)']]
                    y_valid = model.predict(x_valid)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write(np.mean(y_pred))

                    st.subheader('Random Forest Underestimation:')
                    st.image('rf_graph.png')
                    st.write('''
                        Our model found the average of every year's cost and used that to create a prediction through averaging the cost for 2010-2021. This average of 4396.49 contributed to creating the predicted cost of 4396.55 which is less than a dollar difference. We believe that if we had utilized 2021's costs and added a little extra to it then our results would have been more accurate to reality. If we wanted to make a fancy model, probably weight the more recent years higher than the older years and find some creative way to divide the data into training and test sets because we can't really have our model training itself by trying to fit to 2010 costs. However, this okay because we still learned the value behind understanding your data and making sure to map out your thought process behind your model before making it.
                   ''')



                    # new_diamonds["cost (dollars)"] = y_pred
                    # st.subheader('Combined Dataset:')
                    # c_columns = ['carat', 'depth', 'table', 'length (mm)', 'width (mm)', 'height (mm)', 'year', 'cost (dollars)']
                    # c = og_data.columns
                    # og_data = og_data.loc[:, c.isin(c_columns)]
                    # og_data = og_data[c_columns]
                    # joined_df = new_diamonds.append(og_data, ignore_index=True)
                    # st.write(joined_df.shape)
                    # st.write(joined_df.head(5))
                    # st.set_option('deprecation.showPyplotGlobalUse', False)
                    # fig = plt.figure(figsize=(10, 4))
                    # pd.set_option('display.float_format', lambda x: '%.4f' % x)
                    # joined_pivot = joined_df.pivot_table(index='year', values='cost (dollars)', aggfunc='sum',
                    #                                      margins=True)
                    # joined_plot = joined_pivot[:-1].plot()
                    # # plt.plot(joined_df.pivot_table(index='year', values='cost (dollars)', aggfunc='min', margins=True))
                    # plt.ticklabel_format(axis="y", style="sci")
                    # plt.title("Trend of Wholesale Diamond Prices")
                    # plt.ylabel('Cost of Diamonds (in 100 Millions)')
                    # plt.xlabel('Years')
                    # st.write(joined_pivot)
                    # st.pyplot()
                    # st.markdown("""---""")

                    # # cost_names =
                    # st.write("Precision: ", precision_score(y_test, y_pred).round(2))
                    # st.write("Recall: ", recall_score(y_test, y_pred).round(2))
                    #plot_metrics(metrics)

            # html_temp = """
            #   <div style ="background-color:Gray;padding:13px">
            #   <h1 style ="color:black;text-align:center;">Streamlit Diamonds Classifier ML App </h1>
            #   </div>
            #   """
            # # this next line is allowing us to display the frontend aspects that we chose in the previous command
            # st.markdown(html_temp, unsafe_allow_html=True)
            # # the following line show the user what the required data to make the prediction are
            # result = ""
            # # this line will make sure that when people click predict our prediction function is called
            # if st.button("Click here to run the Random Forest Regressor"):
            #     #result = prediction(new_diamonds['carat'],new_diamonds['depth'],new_diamonds['table'],new_diamonds['length (mm)'],new_diamonds['width (mm)'],new_diamonds['height (mm)'])
            #     st.success("The output is {}".format(result))
            #     st.write(result)

    elif uploaded_file is not None and choice == "Prediction using R":

        st.image("LRM1.jpg")
        st.image("LR_MODEL_2.jpg")


    elif choice == "Our Team":
        # st.subheader(choice)
        st.title(" Our Team Members")
        st.write(
            "We are part of the 2022 cohort of Hack Diversityís. Hack Diversityís is tapping into the full potential of the Boston talent landscape by briding the gap in companies' cultures to include high performing talent that predominatly identifies as Black or Latine/x/a/o to launch careers in the technology and innovation realm. Thank you Hack. Diversity for choosing us.")
        st.balloons()
        # col1,mid,col2 =st.columns([1,1,20])
        # with col1:
        #    st.image('team_black.png',use_column_width=False)
        # with col2:
        #    st.write("Eduardo")

        # Another way
        # image = Image.open('team_black.png')
        # size = (800, 800)
        # image.thumbnail(size)
        # fig = plt.figure()
        # plt.imshow(image)
        # plt.axis("off")
        # st.pyplot(fig)

        st.subheader("Hey, I am Eduardo :star:")
        # \n - means next line
        st.markdown('''
              I am Eduardo Sa. I am Business Intelligence student at Umass Boston. I am graduating on May 2022.

              Please check my Social Media:
              - [LinkedIn:](https://www.linkedin.com/in/eduardo-s%C3%A1-73b76286)
              - [Medium:](https://medium.com/hack-diversity-movement/cohort-stories-meet-eduardo-77e1c2805134) \n 

               ''')

        st.subheader("1) Describe what you contributed (briefly) and how that helped the final product?")
        st.markdown('''
       My focus on this project was in the visualization process and in the creation of a framework(streamlit). 
               ''')

        st.subheader("2) Describe how you approached the challenge?")
        st.markdown('''
              In the process of creating visualizations my first approach was to use Tableau in order to get  a sense about the data. Later, I learned with my team about the cleaning process. Since I wanted to be a team player, Iíve installed the required programs and started learning about the python libraries(pandas, numpy, seaborn and others). Later, I heard about streamlit and I decided to learn and implement the knowledge to this project.

               ''')

        st.subheader(
            "3) What was the biggest obstacle you encountered (technically or in collaborating as a team) and how did you overcome it?")
        st.markdown('''
        My biggest obstacle was to code in python using pandas and translate it into streamlit(learning the syntax). I had a lot of bugs in this process. The second challenge creating plots since it works in the jupyter notebook, but it does not work just copying into streamlit. Later, how to make it user friendly and the deployment of it.


               ''')
        st.subheader("4) If you had more time or were starting from the beginning, what would you do differently?")
        st.markdown('''
              I would suggest to our team to apply  Agile methodology in every step of the project.\n 

               ''')

        st.markdown("""---""")

        st.subheader("Hey, I am Cyrus Kirby :star:")
        st.markdown('''
              I am a senior at Tufts University majoring in mathematics and philosophy.\n
              Also check my Social Media:
              - [LinkedIn:](https://www.linkedin.com/in/cyrus-kirby)
              - [Medium:](https://medium.com/hack-diversity-movement/cohort-stories-meet-cyrus-b1210c0eaa9b)
               ''')

        st.subheader("1) Describe what you contributed (briefly) and how that helped the final product?")
        st.markdown('''
         I contributed in finalizing our data cleaning methods, creating a data transformation pipeline, evaluating different models, and had a small role in deploying the finalized model to streamlit. I also served as Ops Lead.
               ''')

        st.subheader("2) Describe how you approached the challenge?")
        st.markdown('''
        I consulted ìHands-On Machine Learning with Scikit-Learn, Kerala, and Tensorflowî for the technical aspects of the project, googled whatever tidbits of information I needed to solve my technical problems, and took some great ideas from my mentor and teammates in shaping the path we took as a group.


               ''')

        st.subheader(
            "3) What was the biggest obstacle you encountered (technically or in collaborating as a team) and how did you overcome it?")
        st.markdown('''
        Sklearnís pipeline functionality is so confusing if youíre not used to it, so even making a simple one took a couple weeks for me. Facilitating communication was also pretty hard, but we eventually settled on slack expectations and I checked in on people very few days.


               ''')
        st.subheader("4) If you had more time or were starting from the beginning, what would you do differently?")
        st.markdown('''
        I would spend a LOT more time data cleaning because we were still having data issues up to the end of February. I would also look into making a model that took the time of sale into account by trying to predict 2021 prices with the previous yearsí data and hypothesizing a small increase in price for 2022. Ops-wise, I would get a better sense of teammatesí motivations and goals in trying to organize project responsibilities and set up clear norms for communication and work early on.\n 

               ''')

        st.markdown("""---""")

        st.subheader("Hey, I am Kevin Troncoso :star:")
        st.markdown('''
             I am a first-generation student studying at Bentley University to become a data analytics professional. A fun fact about me is that I am of Dominican descent and love painting. In terms of what interests me, I enjoy learning about statistics and have a side interest in business intelligence. \n

              Also check my Social Media:
              - [LinkedIn:](https://www.linkedin.com/in/eduardo-s%C3%A1-73b76286)
              - [Medium:](https://medium.com/hack-diversity-movement/cohort-stories-meet-kevin-t-674db5c2dad)
               ''')
        st.subheader("1) Describe what you contributed (briefly) and how that helped the final product?")
        st.markdown('''
         As for my involvement in the project, I helped with determining which model to use for the prediction of wholesale diamond prices and assisted with portions of the streamlit coding. In particular, my main focus was on achieving a plausible/functioning linear model for the dataówhich included transformations and analysis of machine learning termsóand determining its usability and accuracy compared to other models.

               ''')

        st.subheader("2) Describe how you approached the challenge?")
        st.markdown('''
        I approached the challenge by referring back to my previous experience in linear regression. I felt as though it was one of the only models I knew best, and as such, I tried to make an acceptable model. After creating a model that I found suitable, I looked up other models and tried to figure out why and how a linear model wouldnít work, even after transforming it and achieving a high accuracy benchmark. 

               ''')

        st.subheader(
            "3) What was the biggest obstacle you encountered (technically or in collaborating as a team) and how did you overcome it?")
        st.markdown('''
        The biggest obstacle for me was transferring my coding from R to python. I never really had any prior experience with machine learning in python (in comparison to my knowledge in R),  so it took a lot of time to learn sklearn and scikit-learn. 
               ''')
        st.subheader("4) If you had more time or were starting from the beginning, what would you do differently?")
        st.markdown('''
        I would spend more time learning about machine learning/model building in python, that way I could optimize my time in better tune models. I would have also liked to try more models. \n 

               ''')

        st.markdown("""---""")

        st.subheader("Hey, I am Rita Nfamba :star:")

        st.markdown('''
              I'm a Computer Science student at Boston University.\n
              Also check my Social Media:
              - [LinkedIn:](https://www.linkedin.com/in/rita-nfamba-06136511a)
              - [Medium:](https://medium.com/hack-diversity-movement/cohort-stories-meet-rita-c79a0203cc2f)
               ''')

        st.subheader("1) Describe what you contributed (briefly) and how that helped the final product?")
        st.markdown('''
        I contributed towards data cleaning; had to meet a consensus with the team about the values that I didnít understand why they were eliminated from the data set.
        Created  linear regression model purposely  for comparison purposes amongst  my team membersí models. 

               ''')

        st.subheader("2) Describe how you approached the challenge?")
        st.markdown('''
        -I tried to understand the dataset and interpret the different features before I decided on what to do.\n  
        -Checking what type of variables does the data set contain.\n  
        -What type of relationships do I expect to see  between variables.\n  
        -Checking for anomalous behavior. \n  
        -Assessed the different features/variables in the data set to determine what was required to decide on the appropriate requirements for the model(s).\n 

               ''')

        st.subheader(
            "3) What was the biggest obstacle you encountered (technically or in collaborating as a team) and how did you overcome it?")
        st.markdown('''
        My travels away to Botswana, I encountered challenges with communication due to internet instability.  This affected my turn around time and timely delivery of deliverables.\n  


               ''')
        st.subheader("4) If you had more time or were starting from the beginning, what would you do differently?")
        st.markdown('''
         I would do better at time management and commitment.\n 
               ''')

        st.markdown("""---""")

        st.subheader("Hey, I am Justice DelCore :star:")

        st.markdown('''

              I'm a student at the University of South Dakota and an aspiring data scientist.\n
              Also check my Social Media:
              - [LinkedIn:](https://www.linkedin.com/in/delcorej)
              - [Medium:](https://medium.com/hack-diversity-movement/cohort-stories-meet-justice-a67d36f9d718)
               ''')

        st.subheader("1) Describe what you contributed (briefly) and how that helped the final product?")
        st.markdown('''
        I assisted with cleaning the data and doing exploratory data analysis of the diamond dataset. I also assisted Eduardo and Kevin with the implementation of Streamlit. Then once our Streamlit was ready to deploy I deployed it through Google Cloud and the utilization of Google App Engine.
9:04
Normally I would say yes but I am waiting to meet with cyrus to go over some code\n
               ''')

        st.subheader("2) Describe how you approached the challenge?")
        st.markdown('''
        I approached the challenge by stepping in where I was needed. We had a pretty diverse talent range in our team which was awesome to work with. So whenever someone needed assistance with their part or needed someone to delve into making a part stand out I stepped in.

        \n
               ''')

        st.subheader(
            "3) What was the biggest obstacle you encountered (technically or in collaborating as a team) and how did you overcome it?")
        st.markdown('''
        The biggest challenge for us as a team was also our biggest strength. The diverse ranges of knowledge on our team - at first it seemed a difficult aspect because getting everyone on the same level was the original approach but, then we realized that it wasn’t necessary to do so. So instead we let everyone showcase their strengths while also furthering developing the areas they were interested in learning and that’s what really made us make the best project possible.
                \n

               ''')
        st.subheader("4) If you had more time or were starting from the beginning, what would you do differently?")
        st.markdown('''
        If I had more time I would have loved to delve deeper into the machine learning. It’s something that is my weakness in knowledge wise but something that I am so curious about. I really enjoyed seeing my teammates build their models and asking questions about them. But I would have loved to have the time to learn why they chose a specific model or what made a model better than another one.

        \n 
               ''')
        # one way
        # st.image('team_black.png',use_column_width=False)

        # Another way
        image = Image.open('team_black.png')
        size = (800, 800)
        image.thumbnail(size)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)

# The Python is executed directly by the python interpreter
if __name__ == "__main__":
    main()







