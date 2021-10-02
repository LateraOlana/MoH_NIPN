# import numpy as np
# import pandas
# import streamlit as st
# import sklearn
# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# import matplotlib.pyplot as plt
# df = pandas.read_csv("data1.csv")
#
#
# X = df[['ANC','SBA','DPT3','Measles','ARI','Dia','sati','c1','iws','is','eb']]
# y = df['u5']
# # X = df.iloc[:,1:9].values
# # y = df.iloc[:,9].values
# # # # regr = LinearRegression()
# # # # regr.fit(X, y)
# # # poly_reg = PolynomialFeatures(degree=2)
# # # X_poly = poly_reg.fit_transform(X)
# # # lin_reg2 = LinearRegression()
# # # lin_reg2.fit(X_poly,y)
# # # # from sklearn.preprocessing import PolynomialFeatures
# # # #
# # # # poly_reg = PolynomialFeatures(degree=2)
# # # # X_poly = poly_reg.fit_transform(X)
# # # #
# # # # lin_reg2 = linear_model.LinearRegression()
# # # # lin_reg2.fit(X_poly, y)
# # # # X_grid = np.arange(min(X), max(X), 0.1)
# # # # X_grid = X_grid.reshape(len(X_grid), 1)
# # # # plt.scatter(X, y, color='red')
# # # #
# # # # plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
# # # #
# # # # plt.title("Truth or Bluff(Polynomial)")
# # # # plt.xlabel('Position level')
# # # # plt.ylabel('Salary')
# # # # plt.show()
# # # predictedU5 = lin_reg2.predict(poly_reg.fit_transform([[4,1.1,6,4.8,8.8,48.6,7.9,4.5]]))
# # # # predictedU51 = lin_reg2.predict([[84,18.1,60,54.8,84.8,48.6,67.9,94.5]])
# # #
# # # print(lin_reg2.coef_)
# # # print(predictedU5)
# #
# # # X = df.iloc[:,1:8].values
# # # y = df.iloc[:,8].values
# # lin = LinearRegression()
# #
# # lin.fit(X, y)
# # # print(lin.predict([[90,90,76,70,45,63,85,85,75,20,65]]))
# # # Fitting Polynomial Regression to the dataset
# # from sklearn.preprocessing import PolynomialFeatures
# #
# poly = PolynomialFeatures(degree=1)
# X_poly = poly.fit_transform(X)
#
# poly.fit(X_poly, y)
# lin2 = LinearRegression()
# lin2.fit(X_poly, y)
#
# predictedU51=lin2.predict(poly.fit_transform([[73.6+26.4,48+52,61+15,59+15,35.1+16.6,51+19.6,69.1+21,74.3+10.7,67.4+10.6,17.2+17.8,58.8+11]]))
# predictedU52=lin2.predict(poly.fit_transform([[73.6+23.2,48+44,61+10,59+10,35.1+7.6,51+15.2,69.1+14.2,74.3+5,67.4+10.6,17.2+8.8,58.8+5]]))
# predictedU3=lin2.predict(poly.fit_transform([[73.6+2,48+10,61+5,59+5,35.1+5.8,51+8.2,69.1+14.2,74.3+2,67.4+2.6,17.2+1.1,58.8+2.6]]))
#
# print(lin2.score(X_poly, y))
# # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
# # print(mse)
# # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
# print(lin2.coef_)
# print(predictedU51)
# print(predictedU52)
# print(predictedU3)
#
# # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
# # plt.title('', fontsize = 16)
# # plt.ylabel('', fontsize = 12)
# # plt.xlabel('', fontsize = 12)
# #
# # plt.savefig('c.png')
#
import numpy as np
import pandas
import streamlit as st
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import style
from sklearn.preprocessing import PolynomialFeatures
# print(predictedU5_a)
col0 = st.sidebar
col1 = st.sidebar
col2 = st.sidebar

buffer,col3 = st.columns([1,100])
with col0:
    select = st.selectbox('Select Analysis:',('National','Tigray'	,'Afar',	'Amhara',	'Oromia',	'Somali',	'Benishangul-Gumuz',	'SNNP',	'Gambella',	'Harari',	'Dire Dawa',	'Addis Ababa'))
    if select == 'National':
        df = pandas.read_csv("stunting.csv")
        df_w = pandas.read_csv("wasting.csv")
        df_u = pandas.read_csv("underweight.csv")
        df_a = pandas.read_csv("anemia.csv")
        # style.use('seaborn-whitegrid')
        # plt.rcParams['figure.figsize'] = (20, 10)
        # sb.pairplot(df)
        # plt.savefig('pairplor.png')
        # plt.show()

        X = df[['EI', 'CF', 'YCA',  'FBF', 'ASF', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF','IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df['Stunting']
        y_w = df_w['Wasting']
        y_u = df_u['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        var_list_stunting = ['EI', 'CF', 'EBF', 'YCA', 'YCA_MMF', 'FBF', 'ASF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF','IRF', 'VAF', 'WL', 'ISF', 'ISW','Stunting']
        var_list_wasting = ['EI', 'CF', 'EBF', 'YCA', 'YCA_MMF', 'FBF', 'ASF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF','IRF', 'VAF', 'WL', 'ISF', 'ISW','Wasting']
        var_list_underweight = ['EI', 'CF', 'EBF', 'YCA', 'YCA_MMF', 'FBF', 'ASF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF','IRF', 'VAF', 'WL', 'ISF', 'ISW','Underweight']

        importance_stunting = lin2.coef_
        importance_wasting = lin2_w.coef_
        imprortance_underweight = lin2_u.coef_
        var_big_stunting = {}
        var_big_wasting = {}
        var_big_underweight = {}
        # summarize feature importance
        for i, v in enumerate(importance_stunting):
            var_big_stunting[var_list_stunting[i]] = abs(v)
            print('Feature: %0d, Score_stunting: %.5f' % (i, v))
        # plot feature importance
        # plt.bar([x for x in range(len(importance_stunting))], importance_stunting)
        # plt.show()
        for i, v in enumerate(importance_wasting):
            var_big_wasting[var_list_wasting[i]] = abs(v)
            print('Feature: %0d, Score_wasting: %.5f' % (i, v))
        # plot feature importance
        # plt.bar([x for x in range(len(importance_wasting))],  importance_wasting)
        # plt.show()
        for i, v in enumerate(imprortance_underweight):
            var_big_underweight[var_list_underweight[i]] = abs(v)
            print('Feature: %0d, Score_underweight: %.5f' % (i, v))
        # plot feature importance
        # plt.bar([x for x in range(len(imprortance_underweight))],  imprortance_underweight)
        # plt.show()
        # lin2_a.fit(X_poly, y_a)
        sort_orders_stunting = dict(sorted(var_big_stunting.items(), key=lambda x: x[1], reverse=True))
        sort_orders_wasting = dict(sorted(var_big_wasting.items(), key=lambda x: x[1], reverse=True))
        sort_orders_underweight = dict(sorted(var_big_underweight.items(), key=lambda x: x[1], reverse=True))
        print("Stunting")
        print(sort_orders_stunting)
        print("Wasting")
        print(sort_orders_wasting)
        print("Under-weight")
        print(sort_orders_underweight)
        names = list(sort_orders_stunting.keys())
        values = list(sort_orders_stunting.values())

        plt.bar(range(len(sort_orders_stunting)), values, tick_label=names)
        plt.xticks(rotation=90)
        plt.show()
        names = list(sort_orders_wasting.keys())
        values = list(sort_orders_wasting.values())

        plt.bar(range(len(sort_orders_wasting)), values, tick_label=names)
        plt.xticks(rotation=90)
        plt.show()
        names = list(sort_orders_underweight.keys())
        values = list(sort_orders_underweight.values())

        plt.bar(range(len(sort_orders_underweight)), values, tick_label=names)
        plt.xticks(rotation=90)
        plt.show()
        print("results")
        #2024 p1
        # predictedU5 = lin2.predict(poly.fit_transform([[82.0,80.0,77.0,50.0,65.0,17.0,23.1,93.3,35.0,38.0,29.0,87.5,26.2,40.0,55.0,59.6,39.5,75.0]]))
        # predictedU5_w = lin2_w.predict(poly_w.fit_transform([[82.0,80.0,77.0,50.0,65.0,17.0,23.1,93.3,35.0,38.0,29.0,87.5,26.2,40.0,55.0,59.6,39.5,75.0]]))
        # predictedU5_u = lin2_u.predict(poly_u.fit_transform([[82.0,80.0,77.0,50.0,65.0,17.0,23.1,93.3,35.0,38.0,29.0,87.5,26.2,40.0,55.0,59.6,39.5,75.0]]))

        # #2024 p2
        # predictedU5 = lin2.predict(poly.fit_transform([[84.0,85.0,80.0,52.0,67.0,17.0,31.0,93.3,43.0,46.0,37.0,89.0,34.2,48.0,63.0,63.6,47.5,77.0]]))
        # predictedU5_w = lin2_w.predict(poly_w.fit_transform([[84.0,85.0,80.0,52.0,67.0,17.0,31.0,93.3,43.0,46.0,37.0,89.0,34.2,48.0,63.0,63.6,47.5,77.0]]))
        # predictedU5_u = lin2_u.predict(poly_u.fit_transform([[84.0,85.0,80.0,52.0,67.0,17.0,31.0,93.3,43.0,46.0,37.0,89.0,34.2,48.0,63.0,63.6,47.5,77.0]]))

        # # 2030 p3
        # predictedU5 = lin2.predict(poly.fit_transform([[100.0,98.0,95.0,75.0,90.0,35.0,73.1,100.0,85.0,82.0,73.0,100.0,64.2,90.0,100.0,83.6,82.5,100.0]]))
        # predictedU5_w = lin2_w.predict(poly_w.fit_transform([[100.0,98.0,95.0,75.0,90.0,35.0,73.1,100.0,85.0,82.0,73.0,100.0,64.2,90.0,100.0,83.6,82.5,100.0]]))
        # predictedU5_u = lin2_u.predict(poly_u.fit_transform([[100.0,98.0,95.0,75.0,90.0,35.0,73.1,100.0,85.0,82.0,73.0,100.0,64.2,90.0,100.0,83.6,82.5,100.0]]))

        # # 2030 p2
        # predictedU5 = lin2.predict(poly.fit_transform([[90.0,90.0,85.0,70.0,85.0,35.0,53.1,100.0,65.0,62.0,53.0,97.0,46.2,70.0,85.0,74.6,64.5,90.0]]))
        # predictedU5_w = lin2_w.predict(poly_w.fit_transform([[90.0,90.0,85.0,70.0,85.0,35.0,53.1,100.0,65.0,62.0,53.0,97.0,46.2,70.0,85.0,74.6,64.5,90.0]]))
        # predictedU5_u = lin2_u.predict(poly_u.fit_transform([[90.0,90.0,85.0,70.0,85.0,35.0,53.1,100.0,65.0,62.0,53.0,97.0,46.2,70.0,85.0,74.6,64.5,90.0]]))

        # 2030 p1
        predictedU5 = lin2.predict(poly.fit_transform([[86.0,87.0,83.0,60.0,70.0,25.0,40.0,93.3,50.0,53.0,43.0,91.0,37.2,55.0,70.0,65.6,52.5,81.0]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[86.0,87.0,83.0,60.0,70.0,25.0,40.0,93.3,50.0,53.0,43.0,91.0,37.2,55.0,70.0,65.6,52.5,81.0]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[86.0,87.0,83.0,60.0,70.0,25.0,40.0,93.3,50.0,53.0,43.0,91.0,37.2,55.0,70.0,65.6,52.5,81.0]]))

        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 47.4, 100.0)
            number4 = st.slider("Complimentary Food(CF)", 36.5, 100.0)
            # number5 = st.slider("Exclusively Breastfeding(EBF)", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 1.0, 100.0)
            # number7 = st.slider("Minimum Meal Frequency (MMF)", 41.8, 100.0)
            number8 = st.slider("Fortified baby food (FBF)", 3.1, 100.0)
            number9 = st.slider("Children Receiving Meat, Fish, Poultry(ASF)", 0.0, 100.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.6, 100.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 4.0, 100.0)
        with col2:
            # number11 = st.slider("Iodized Salt in Household(ISIHH)", 25.5, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 6.4, 100.0)
            number13 = st.slider("Family Planning(FP)", 5.9, 100.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 4.7, 100.0)
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 18.1, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.3, 100.0)
            # number17 = st.slider("Household with Sanitation Facility", 17.4, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 6.4, 100.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.3, 100.0)
            number20 = st.slider("Women Literate(WL)", 24.4, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(poly.fit_transform([[number3, number4, number6,  number8,
                                                            number9,  number12, number13, number14, number15,
                                                            number16,  number18, number19, number20, number10,
                                                            number1]]))
            predictedU5_w = lin2_w.predict(poly_w.fit_transform([[number3, number4,  number6,  number8,
                                                                  number9,  number12, number13, number14,
                                                                  number15, number16,  number18, number19,
                                                                  number20, number10, number1]]))
            predictedU5_u = lin2_u.predict(poly_u.fit_transform([[number3, number4,  number6, number8,
                                                                  number9,  number12, number13, number14,
                                                                  number15, number16, number18, number19,
                                                                  number20, number10, number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Tigray':
        df_t = pandas.read_csv("Tigray.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40,20.20,63.50,87.70,38.90,26.80,39.90,69.00,14.20,19.10,36.10,52.70,59.60,33.00,85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40,20.20,63.50,87.70,38.90,26.80,39.90,69.00,14.20,19.10,36.10,52.70,59.60,33.00,85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40,20.20,63.50,87.70,38.90,26.80,39.90,69.00,14.20,19.10,36.10,52.70,59.60,33.00,85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 28.4, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 1.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 9.6, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 1.3, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 5.1, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 22.3, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 11.8, 70.0)
            number13 = st.slider("Family Planning(FP)", 6.9, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 6.7, 70.0)
        with col2:

            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 28.50, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.3, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 10.7, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 1.9, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 1.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 22.6, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                                            number16, number17, number18, number19, number20, number10,
                                                            number1]]))
            predictedU5_w = lin2_w.predict(poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                                            number16, number17, number18, number19, number20, number10,
                                                            number1]]))
            predictedU5_u = lin2_u.predict(poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                                            number16, number17, number18, number19, number20, number10,
                                                            number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Afar':
        df_t = pandas.read_csv("Afar.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)

        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Oromia':
        df_t = pandas.read_csv("Oromia.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Amhara':
        df_t = pandas.read_csv("Amhara.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Somali':
        df_t = pandas.read_csv("Somali.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Benishangul-Gumuz':
        df_t = pandas.read_csv("Ben.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'SNNP':
        df_t = pandas.read_csv("snnp.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Gambella':
        df_t = pandas.read_csv("gambela.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))

            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Harari':
        df_t = pandas.read_csv("harari.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Dire Dawa':
        df_t = pandas.read_csv("diredawa.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
    if select == 'Addis Ababa':
        df_t = pandas.read_csv("Addis_Ababa.csv")

        X = df_t[['EI', 'YCA', 'YCA_MMF', 'ISH', 'FV', 'FP', 'IWS', 'WOP_L30', 'HH_BSF', 'HSF',
                  'IRF', 'VAF', 'WL', 'ISF', 'ISW']]
        y = df_t['Stunting']
        y_w = df_t['Wasting']
        y_u = df_t['Underweight']
        poly = PolynomialFeatures(degree=1)
        poly_w = PolynomialFeatures(degree=1)
        poly_u = PolynomialFeatures(degree=1)
        # poly_a = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        poly_w.fit(X_poly, y_w)
        poly_u.fit(X_poly, y_u)
        # poly_a.fit(X_poly, y_a)
        lin2 = LinearRegression()
        lin2_w = LinearRegression()
        lin2_u = LinearRegression()
        lin2_a = LinearRegression()
        lin2.fit(X_poly, y)
        lin2_w.fit(X_poly, y_w)
        lin2_u.fit(X_poly, y_u)
        # lin2_a.fit(X_poly, y_a)
        predictedU5 = lin2.predict(poly.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00, 14.20,
                                                        19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_w = lin2_w.predict(poly_w.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        predictedU5_u = lin2_u.predict(poly_u.fit_transform([[73.40, 20.20, 63.50, 87.70, 38.90, 26.80, 39.90, 69.00,
                                                              14.20, 19.10, 36.10, 52.70, 59.60, 33.00, 85.60]]))
        # predictedU5_a = lin2_a.predict(poly_a.fit_transform([[72.00,69.00,58.80,13.50,55.10,4.80,3.90,89.30,18.20,28.80,19.80,56.90,10.20,53.40,24.20,39.40,47.60]]))
        print(predictedU5)
        print(predictedU5_w)
        print(predictedU5_u)
        with col1:
            number3 = st.slider("Ealy Initiation(EI)", 0.0, 100.0)
            # number4 = st.slider("CF", 36.5, 100.0)
            # number5 = st.slider("EBF", 49.0, 100.0)
            number6 = st.slider("Children 6-23 months fed 5+ food groups(5+FG)", 0.0, 50.0)
            number7 = st.slider("Minimum Meal Frequency (MMF)", 0.0, 100.0)
            # number8 = st.slider("FBF", 3.1, 20.0)
            # number9 = st.slider("ASF", 0.0, 20.0)
            number10 = st.slider("Improved Sanitation Facility(ISF)", 0.0, 60.0)
            number1 = st.slider("Iron Supplement for Women(ISW)", 0.0, 100.0)
            number11 = st.slider("Iodized Salt in Household(ISIHH)", 0.0, 100.0)
            number12 = st.slider("Fully Vaccinated(FV)", 0.0, 70.0)
            number13 = st.slider("Family Planning(FP)", 0.0, 70.0)
            number14 = st.slider("Households with an Improved Water Source(IWS) ", 0.0, 70.0)
        with col2:
            number15 = st.slider("Water and Sanitation: Source of drinking water(WOP_L30)", 0.0, 100.0)
            number16 = st.slider("Households with Basic Sanitation Facility(HH_BSF)", 0.0, 20.0)
            number17 = st.slider("Household with Sanitation Facility", 0.0, 100.0)
            number18 = st.slider("Iron Rich Foods(IRF)", 0.0, 70.0)
            number19 = st.slider("Vitamin A Foods(VAF)", 0.0, 100.0)
            number20 = st.slider("Women Literate(WL)", 0.0, 100.0)
        with col3:
            import altair as alt
            import pandas as pd

            predictedU5 = lin2.predict(
                poly.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                     number16, number17, number18, number19, number20, number10,
                                     number1]]))
            predictedU5_w = lin2_w.predict(
                poly_w.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            predictedU5_u = lin2_u.predict(
                poly_u.fit_transform([[number3, number6, number7, number11, number12, number13, number14, number15,
                                       number16, number17, number18, number19, number20, number10,
                                       number1]]))
            if predictedU5[0] <= 0:
                predictedU5[0] = 'NaN'
            if predictedU5_w[0] <= 0:
                predictedU5_w[0] = 'NaN'
            if predictedU5_u[0] <= 0:
                predictedU5_u[0] = 'NaN'
            # predictedU5_a=lin2_a.predict(poly_a.fit_transform([[number3,number4,number5,number6,number7,number8,number9,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20]]))
            # print(lin2.score(X_poly, y))
            # mse = sklearn.metrics.mean_squared_error([30], [predictedU5])
            # print(mse)
            # predictedU5=lin2.predict(poly.fit_transform([[76,13.3,50,52.3,79.4,41.55,62,83.5]]))
            # print(lin2.coef_)
            # print(predictedU5)

            # plt.scatter(x = 'SBA', y = 'u5', data = df, s = 100, alpha = 0.9, edgecolor = 'white')
            # plt.title('', fontsize = 16)
            # plt.ylabel('', fontsize = 12)
            # plt.xlabel('', fontsize = 12)
            #
            # plt.savefig('c.png')
            source = pd.DataFrame({
                'a': ['Stunting', 'Wasting', 'Under-weight'],
                'b': [predictedU5[0], predictedU5_w[0], predictedU5_u[0]]
            })

            bars = alt.Chart(source).mark_bar().encode(
                x='a',
                y='b'
            )

            text = bars.mark_text(
                align='right',
                baseline='middle'
            ).encode(
                text='a'
            )
            (bars + text).properties(height=5000)
            st.altair_chart(bars, use_container_width=True)
            st.write("Stunting: ", predictedU5[0])
            st.write("Wasting: ", predictedU5_w[0])
            st.write("Under-weight: ", predictedU5_u[0])
            # st.write("Anemia-child: ", predictedU5_a[0])
st.write("If the result of the simulation is NaN, that means the projected value is close zero.")