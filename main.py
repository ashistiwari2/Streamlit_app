import streamlit as st
import pandas as pd
from time import sleep
from sklearn.linear_model import LogisticRegression
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
file_csv=pd.read_csv('Fish.csv')
x=file_csv.iloc[:,1:].values
y=file_csv.iloc[:,0:-6].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

def dataframe(data):
    index = ['1', '2', '3', '4', '5', '6', '7', '8']
    df = pd.DataFrame(data,
                      columns=('model', 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width',
                               'Species','accuracy score'), index=index)
    df=df.drop_duplicates()
    df.to_csv('prediction_data.csv', index=False)
    df.to_excel.xls.writer('prediction_data.xls', index=False)
    st.dataframe(df)

def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      .stApp {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
           background-size: cover
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )
def main():
    st.title('Machine learning model')
    st.markdown('Using Logistic regression and random forest')
    side_bg="download.jpg"
    sidebar_bg(side_bg)
    model = st.selectbox('Select the model', ('logistic_regression', 'random_forest'))
    if model =='logistic_regression':
        st.subheader('Using logistic regression')
        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        with st.form(key='logistic_regression', clear_on_submit=True):
            Weight = st.number_input("Enter weight of fish", key='w_1',value=1.0)
            Length1=st.number_input("Enter length1 of fish",key='l_1',value=2.0)
            Length2 = st.number_input("Enter length2 of fish", key='l_2',value=3.0)
            Length3=st.number_input("Enter length3 of fish",key='l_3',value=4.0)
            Height=st.number_input("Enter Height of fish",key='h_1',value=5.0)
            Width=st.number_input("Enter width of fish",key='w_2',value=6.0)
            submit_text = st.form_submit_button(label='Submit')
            a=[Weight,Length1,Length2,Length3,Height,Width]
            if submit_text:
                col1,col2=st.columns(2)
                with st.spinner(f'analysing with given data:{a}.....'):
                    sleep(10)
                    with col1:
                        training_data = [[
                            float(i) for i in a
                        ]]
                        # print(training_data)
                        class_idx = clf.predict(training_data)[0]
                        st.subheader('Fish species :{}'.format(class_idx))
                    with col2:
                        y_pred = clf.predict(X_test)
                        print(y_pred)
                        score = accuracy_score(Y_test, y_pred)
                        st.subheader("Acuuracy score:{}".format(score))

                data=[[model,Weight,Length1,Length2,Length3,Height,Width,class_idx,score]]
                dataframe(data)



    elif model=='random_forest':
        st.subheader('using Random Forest classifier')
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train,Y_train)
        with st.form(key='logistic_regression', clear_on_submit=True):
            Weight = st.number_input("Enter weight of fish", key='w_1_1',value=1.0)
            Length1=st.number_input("Enter length1 of fish",key='l_1_1',value=2.0)
            Length2 = st.number_input("Enter length2 of fish", key='l_2_1',value=3.0)
            Length3=st.number_input("Enter length3 of fish",key='l_3_1',value=4.0)
            Height=st.number_input("Enter Height of fish",key='h_1_1',value=5.0)
            Width=st.number_input("Enter width of fish",key='w_2_1',value=6.0)
            submit_text = st.form_submit_button(label='Submit')
            a=[Weight,Length1,Length2,Length3,Height,Width]
            if submit_text:
                col1,col2=st.columns(2)
                with st.spinner(f'analysing with given data:{a}.....'):
                    sleep(10)
                    with col1:
                        training_data = [[
                            float(i) for i in a
                        ]]
                        print(training_data)
                        class_idx = clf.predict(training_data)[0]
                        st.subheader('Fish species :{}'.format(class_idx))

                    with col2:
                        y_pred = clf.predict(X_test)
                        score = accuracy_score(Y_test, y_pred)
                        st.subheader("Acuuracy score:{}".format(score))

                data = [[model, Weight, Length1, Length2, Length3, Height, Width, class_idx,score]]
                dataframe(data)





if __name__ == '__main__':
    main()







