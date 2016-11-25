#encoding=utf8

import os
import FukuML.Utility as utility
import FukuML.SupportVectorMachine as svm

input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'library/iNDIEVOX-Dataset/dataset/emotion_combine_song_train.dataset')

svm_mc = svm.MultiClassifier()
svm_mc.load_train_data(input_train_data_file)
svm_mc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=1)
svm_mc.init_W()
svm_mc.train()

print("W 平均錯誤值（Ein）：")
print(svm_mc.calculate_avg_error_all_class(svm_mc.train_X, svm_mc.train_Y, svm_mc.W))

cross_validator = utility.CrossValidator()

svm_mc1 = svm.MultiClassifier()
svm_mc1.load_train_data(input_train_data_file)
svm_mc1.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=1)
svm_mc2 = svm.MultiClassifier()
svm_mc2.load_train_data(input_train_data_file)
svm_mc2.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=10)
svm_mc3 = svm.MultiClassifier()
svm_mc3.load_train_data(input_train_data_file)
svm_mc3.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=100)

print("\n10 fold cross validation：")

cross_validator.add_model(svm_mc1)
cross_validator.add_model(svm_mc2)
cross_validator.add_model(svm_mc3)
avg_errors = cross_validator.excute()

print("\n各模型驗證平均錯誤：")
print(avg_errors)
print("\n最小平均錯誤率：")
print(cross_validator.get_min_avg_error())

print("\n取得最佳模型：")
best_model = cross_validator.get_best_model()
print(best_model)

best_model.init_W()
best_model.train()

future_data = '0.0333526059402 0.0191216300488 3.03812655065 0.127020592266 0.190156354743 0.189344178186 0.00923871449698 0.0369026360544 -27.1320427145 2.66185836951 0.0217216187181 0.267235209107 0.0447421589098 0.0781276071067 -0.0571366559669 0.103126054838 0.119812204541 0.083237330401 0.0554364332019 0.0552852970445 0.0626760752385 0.0139713503116 0.00194812234629 0.0238575954008 0.0020666783702 0.00661250790514 0.070888928901 0.000554377872914 0.004143435992 0.0175680180412 0.00153362621193 0.0780878700046 0.00357780308724 0.036753814463 0.0241704507144 0.0140516652302 0.159415283788 0.0381936670858 0.0312071640758 0.186192235083 0.00800724717229 0.0361140229755 1.55024409113 0.551328997095 0.479948068875 0.3745333661 0.328184093716 0.316824345812 0.32149665579 0.316198376933 0.317540592418 0.310620705749 0.306342267441 0.269425248353 0.232773251735 0.0160563041867 0.00261213379927 0.0215200126771 0.00321391292813 0.0089463275766 0.0408386206878 0.000901063838026 0.00607051205129 0.0179243605362 0.0019691103382 0.0419804310717 0.00518539335929 0.0087182119318'
prediction = best_model.prediction(future_data, mode='future_data')
print(prediction)

input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'library/iNDIEVOX-Dataset/dataset/emotion_combine_song_test.dataset')
best_model.load_test_data(input_test_data_file)
print("W 平均錯誤率（Ein）：")
print(best_model.calculate_avg_error_all_class(best_model.train_X, best_model.train_Y, best_model.W))
print("W 平均錯誤率（Eout）：")
print(best_model.calculate_avg_error_all_class(best_model.test_X, best_model.test_Y, best_model.W))
