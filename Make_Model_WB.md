

    %load_ext autoreload

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



    %autoreload 2


    from app.email_class_model import *


    X, y = main()

    done in -1.076s.



    X.shape, y.shape




    ((884, 5766), (884,))




    X_train, X_test, y_train, y_test = create_datasets(X, y)


    X_train.shape, X_test.shape, y_train.shape, y_test.shape




    ((618, 5766), (266, 5766), (618,), (266,))




    # Number of False and Trues in the y_test
    np.bincount(y_test) 




    array([225,  41])




    # Number of False and Trues in the y_train
    np.bincount(y_train) 




    array([542,  76])



# LogisticRegression

Discriminative - 
prob of label given a word - conditional top down 
Works well for this problem because binary classification model needed (if not then need 1 vs all setup)
Only model target variable conditionl on the observed variaables

3/20 - Running lr in ipython to double check the outcome of the actual model


    lr = LogisticRegression(penalty='l1')


    lr.fit( X_train, y_train)




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)




    lr.score(X_test, y_test)




    0.83834586466165417




    lr.coef_ # position in array corresponds to the position in vector




    array([ 0.,  0.,  0., ...,  0.,  0.,  0.])




    y_pred = lr.predict(X_test)


    float(([y_val == y_pred[i] for i, y_val in np.ndenumerate(y_test)]).count(True)) / len(y_pred)




    0.8383458646616542




    # Checking out GridSearch
    
    # get approximate best C value for logistic regression
    params = {'C': np.logspace(0, 9, 10)}
    alg = LogisticRegression()
    clf = GridSearchCV(alg, params, cv=3, n_jobs=-1)
    %time clf.fit(x_train, y_train)
    approx_best_c = clf.best_params_['C']


    # 3/21 Run
    lr_model = build_model(LogisticRegression(penalty='l1'), X_train, y_train)
    predict_eval(lr_model, X_test, y_test)

    Straight Accuracy Calc: 0.845864661654
    SKLearn Accuracy Score: 0.84586
    Confusion Matrix: 
    [[  1  40]
     [  1 224]]



![png](Make_Model_WB_files/Make_Model_WB_19_1.png)


    fpr [ 0.          0.00444444  1.        ] tpr [ 0.          0.02439024  1.        ]



![png](Make_Model_WB_files/Make_Model_WB_19_3.png)



    plot_cross_val(lr_model, X_test, y_test)

    Min Score = 0.486667
    Mean Score = 0.517631
    Max Score = 0.553846



![png](Make_Model_WB_files/Make_Model_WB_20_1.png)


Practice pickling model and proved that it works


    pickle_stuff(lr_model, './model_pkl/lr_321.pkl')


    lr_model_pkl = unpickle_stuff('./model_pkl/lr_321.pkl')


    predict_eval(lr_model_pkl, X_test, y_test)

    Straight Accuracy Calc: 0.845864661654
    SKLearn Accuracy Score: 0.84586
    Confusion Matrix: 
    [[224   1]
     [ 40   1]]



![png](Make_Model_WB_files/Make_Model_WB_24_1.png)


# Naive Bayes

Generative
generate the prediction based on previous - converge to best error faster than discriminative
prob of word and label - join bottom up approach to classificaiton
generate probabilities for all variables
simulate values of variables in a model

Gaussian - assumes normal and sklearn doesn't work well with sparse - good for continuous data


    # 3/21 Run - want 0 on false negatives captured, ok to have false positives
    # TP - sensitivity FN
    # FP - specificity TN
    
    g_nb_model = build_model(GaussianNB(), X_train.todense(), y_train)
    predict_eval(g_nb_model, X_test.todense(), y_test)

    Straight Accuracy Calc: 0.906015037594
    SKLearn Accuracy Score: 0.90602
    Confusion Matrix: 
    [[ 27  14]
     [ 11 214]]



![png](Make_Model_WB_files/Make_Model_WB_28_1.png)


    fpr [ 0.          0.04888889  1.        ] tpr [ 0.          0.65853659  1.        ]



![png](Make_Model_WB_files/Make_Model_WB_28_3.png)



    plot_cross_val(g_nb_model, X_test.todense(), y_test)

    Min Score = 0.560476
    Mean Score = 0.674542
    Max Score = 0.772381



![png](Make_Model_WB_files/Make_Model_WB_29_1.png)


Multinomial - more complex regarding math but works on sparse matrix (initial pass got .84211 Accuracy and same if changed to todense)


    # 3/21 Run
    m_nb_model = build_model(MultinomialNB(), X_train, y_train)
    predict_eval(m_nb_model, X_test, y_test)

    Straight Accuracy Calc: 0.845864661654
    SKLearn Accuracy Score: 0.84586
    Confusion Matrix: 
    [[  0  41]
     [  0 225]]



![png](Make_Model_WB_files/Make_Model_WB_31_1.png)


    fpr [ 0.  1.] tpr [ 0.  1.]



![png](Make_Model_WB_files/Make_Model_WB_31_3.png)



    plot_cross_val(m_nb_model, X_test, y_test)

    Min Score = 0.598974
    Mean Score = 0.640928
    Max Score = 0.691429



![png](Make_Model_WB_files/Make_Model_WB_32_1.png)


Bernoulli - need to binarize the data to make this work (1s and 0s)


    X_train_bin = Binarizer().fit_transform(X_train)
    y_train_bin = Binarizer().fit_transform(y_train)
    X_test_bin = Binarizer().fit_transform(X_test)
    y_test_bin = Binarizer().fit_transform(y_test)


    # 3/21 Run
    b_nb_model = build_model(BernoulliNB(), X_train_bin, y_train_bin)
    predict_eval(b_nb_model, X_test_bin, y_test_bin)

    Straight Accuracy Calc: 0.857142857143
    SKLearn Accuracy Score: 0.85714
    Confusion Matrix: 
    [[ 21  20]
     [ 18 207]]



![png](Make_Model_WB_files/Make_Model_WB_35_1.png)


    fpr [ 0.    0.08  1.  ] tpr [ 0.          0.51219512  1.        ]



![png](Make_Model_WB_files/Make_Model_WB_35_3.png)



    plot_cross_val(b_nb_model, X_test, y_test)

    Min Score = 0.631905
    Mean Score = 0.737009
    Max Score = 0.844762



![png](Make_Model_WB_files/Make_Model_WB_36_1.png)



    

# SVM

Max margin classifier that tries to find separating hyperplane by maximize distance to the points on both sides
One downside to SVMs is that they are often sensitive to changes in hyperparameters or the dataset they were trained on, and do not handle unbalanced classes.
Take too long to train


    # 3/21 Run - Standard Scalar needed? - had to add probability to enable probability estimates to get the cross val to work - what is this?
    svc_model = build_model(SVC(probability=True), X_train, y_train)
    predict_eval(svc_model, X_test, y_test)

    Straight Accuracy Calc: 0.845864661654
    SKLearn Accuracy Score: 0.84586
    Confusion Matrix: 
    [[  0  41]
     [  0 225]]



![png](Make_Model_WB_files/Make_Model_WB_40_1.png)



![png](Make_Model_WB_files/Make_Model_WB_40_2.png)



     # 3/21 Run - SVR and found it will only take it if y is not binary / integer



    plot_cross_val(svc_model, X_test, y_test)

    Min Score = 0.608571
    Mean Score = 0.732161
    Max Score = 0.810476



![png](Make_Model_WB_files/Make_Model_WB_42_1.png)


# Random Forest


    #3/21 Run - dense matrix is required
    rf_model = build_model(RandomForestClassifier(), X_train.todense(), y_train)
    predict_eval(rf_model, X_test.todense(), y_test)

    Straight Accuracy Calc: 0.906015037594
    SKLearn Accuracy Score: 0.90602
    Confusion Matrix: 
    [[ 17  24]
     [  1 224]]



![png](Make_Model_WB_files/Make_Model_WB_44_1.png)



![png](Make_Model_WB_files/Make_Model_WB_44_2.png)



    plot_cross_val(rf_model, X_test.todense(), y_test)

    Min Score = 0.736667
    Mean Score = 0.796520
    Max Score = 0.885714



![png](Make_Model_WB_files/Make_Model_WB_45_1.png)



    

# Gradient Boost

Advantages of GBRT are:
Natural handling of data of mixed type (= heterogeneous features)
Predictive power
Robustness to outliers in input space (via robust loss functions)
The disadvantages of GBRT are:
Scalability, due to the sequential nature of boosting it can hardly be parallelized.


    #3/21 Run dense matrix is required
    gb_model = build_model(GradientBoostingClassifier(), X_train.todense(), y_train)
    predict_eval(gb_model, X_test.todense(), y_test)

    Straight Accuracy Calc: 0.913533834586
    SKLearn Accuracy Score: 0.91353
    Confusion Matrix: 
    [[ 20  21]
     [  2 223]]



![png](Make_Model_WB_files/Make_Model_WB_49_1.png)



![png](Make_Model_WB_files/Make_Model_WB_49_2.png)



    plot_cross_val(gb_model, X_test.todense(), y_test)

    Min Score = 0.782381
    Mean Score = 0.843248
    Max Score = 0.937619



![png](Make_Model_WB_files/Make_Model_WB_50_1.png)



    

# Ada Boost Classifier


    


    #3/21 Run dense matrix is required
    ab_model = build_model(AdaBoostClassifier(), X_train.todense(), y_train)
    predict_eval(ab_model, X_test.todense(), y_test)

    Straight Accuracy Calc: 0.906015037594
    SKLearn Accuracy Score: 0.90602
    Confusion Matrix: 
    [[ 19  22]
     [  3 222]]



![png](Make_Model_WB_files/Make_Model_WB_54_1.png)



![png](Make_Model_WB_files/Make_Model_WB_54_2.png)



    plot_cross_val(ab_model, X_test.todense(), y_test)

    Min Score = 0.780952
    Mean Score = 0.856996
    Max Score = 0.927179



![png](Make_Model_WB_files/Make_Model_WB_55_1.png)



    
