import sys
import random
import math


#### FUNCTIONS #########

#######################
##### Dot Product
######################
def dotproduct(u, v) :
    #    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0,no_of_cols_traindata, 1) :
        dp += u[i] * v[i]
    return dp


######################
##### Standardize data
######################

def standardize_data(traindata, testdata) :
    # traindata
    list1 = []
    for j in range(len(traindata[0])) :
        sum = 0
        for i in range(len(traindata)) :
            sum = sum + (traindata[i][j] ** 2)
        sum = (sum ** 0.5)
        list1.append(sum)
    list2 = []
    for i in range(len(traindata)) :
        list3 = []
        for j in range(len(list1)) :
            if list1[j] == 0 :
                list3.append(0)
            else :
                list3.append(traindata[i][j] / list1[j])
        list2.append(list3)

    list4 = []
    for j in range(len(testdata[0])) :
        sum = 0
        for i in range(len(testdata)) :
            sum = sum + (testdata[i][j] ** 2)
        sum = (sum ** 0.5)
        list4.append(sum)
    list6 = []
    for i in range(len(testdata)) :
        list5 = []
        for j in range(len(list4)) :
            if list4[j] == 0 :
                list5.append(0)
            else :
                list5.append(testdata[i][j] / list4[j])
        list6.append(list5)

    return [list2, list6]


######################
###Least Squares
######################

def least_squares(traindata, trainlabels) :
    # compute slope
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x_sq = 0
    sum_y_sq = 0
    n = 0

    x = []
    y = []
    xy = []
    for i in range(0, no_of_cols_traindata, 1) :
        x.append(1)
        y.append(1)
        xy.append(1)
    for i in range(0, no_of_rows_traindata, 1) :
        if (trainlabels[i]!=0):
            n += 1
            for j in range(0, no_of_cols_traindata, 1) :
                x[j] += traindata[i][j]
                sum_x += traindata[i][j]
                sum_x_sq += traindata[i][j] * traindata[i][j]
        if(trainlabels[i] != 0):
            for j in range(0, no_of_cols_traindata, 1) :
                y[j] += traindata[i][j]
                sum_y += traindata[i][j]
                sum_y_sq += traindata[i][j] * traindata[i][j]
    for j in range(0, len(x)) :
        xy.append(x[j] * y[j])
    for ele in range(0, len(xy)) :
        sum_xy += xy[ele]
    # calculate slope m
    w0 = (n * sum_xy - (sum_x * sum_y)) / ((n * sum_x_sq) - (sum_x_sq))
    print("NOW COMPUTING LEAST SQUARES...." '\n')
    print("w0= ", w0)
    # calculate y-intercept
    w = 0
    w = ((sum_y) - w0 * sum_x) / n
    print("w = ", w)
    print('\n \n \n')
    return w, w0


# Least_squares_regularized
def least_squares_regularized(traindata, trainlabels) :
    w = []
    lamb2 = 0.001
    for j in range(0, len(traindata[0]), 1) :
        # print(random.random())
        w.append(0.02 * random.random() - 0.01)
    # print(len(w))

    # print(w)
    eta = 0.001
    error = len(traindata) + 10
    diff = 1
    count = 0
    # for i in range(0, 1500, 1):
    while ((diff) > 0.000001) :
        dellf = []
        for m in range(0, len(traindata[0]), 1) :
            dellf.append(0)
        for j in traindata :
            if j in trainlabels !=None :
            # if (trainlabels[j]) != None :
                dp = dotproduct(w, traindata[j])
                for k in range(0, len(traindata[0]), 1) :
                    dellf[k-1] += (-(trainlabels[j] - dp) * traindata[j][k] + (w[k] * lamb2)) * 2
                    break
        for j in range(0, len(traindata[0]), 1) :
            w[j] = w[j] - eta * dellf[j]
            reg = lamb2 * (dotproduct(w, w))
        prev = error
        error = 0
        for j in range(0, len(traindata), 1) :
            if j in trainlabels !=None :
            #if (trainlabels[j] != None) :
                # print(dot_product(w,data[j]))
                error += (trainlabels[j] - dotproduct(w, traindata[j])) ** 2
        error += reg
        if (prev > error) :
            diff = prev - error
        else :
            diff = error - prev
        count = count + 1
        if (count % 100 == 0) :
            print(error)
            # print(dellf)

    print('NOW COMPUTING REGULARIZED LEAST SQUARES.....', '\n')
    print("w0 " + str(error))
    print('w: ', w[0:2])
    print('\n \n \n')

    return w, error


###################
######Hinge Loss
###################
def hinge_loss(traindata, trainlabels) :
    # Initialize w
    w = []
    for j in range(0, no_of_cols_traindata, 1) :
        w.append(float(0.02 * random.uniform(0, 1) - 0.01))
    # dellf descent iteration
    eta = 0.01
    diff = 1
    error = 0
    stop = 0.01
    count = 0

    # compute dellf and error
    while (diff > stop) :
        dellf = []
        dellf.extend(0 for _ in range(len(traindata[0])))

        for i in range(0, no_of_rows_traindata, 1) :
            if i in trainlabels !=None :
                a = trainlabels[i] * dotproduct(w, traindata[i])
                for j in range(0, no_of_cols_traindata) :
                    if a < 1 :
                        dellf[j] += -(trainlabels[i] * traindata[i][j])
                    else :
                        dellf[j] += 0

        # update w
        for j in range(0, no_of_cols_traindata, 1) :
            w[j] -= eta * dellf[j]
        prevObj = error
        error = 0
        # compute error

        for i in range(0, no_of_cols_traindata, 1) :
            if i in trainlabels !=None :
                error += max(0, 1 - (trainlabels.get(i)) * dotproduct(w, traindata[i]))
                break
        diff = abs(prevObj - error)
    print('NOW COMPUTING HINGE LOSS....','\n')
    print('w =',w[0:2])

    # distance from origin calculation
    normw = 0
    for j in range(0, no_of_cols_traindata - 1, 1) :
        normw += w[j] ** 2
    print('w0 =',w[2])
    print('\n')
    print('\n \n \n')
    return w,normw

###################
###### Regularized Hinge Loss
###################

def hinge_loss_regularized(traindata, trainlabels) :
    w = []
    for i in range(no_of_rows_traindata) :
        w.append(0)
    for j in range(no_of_rows_traindata) :
        w[j] = 0.02 * random.random() - 0.01
    eta = 0.01
    stop = 0.01
    error = 10
    temp = 0
    while True :
        temp = abs(error)
        error = 0
        c = 0.01
        dellf = []
        normw = 0
        for j in range(0, no_of_cols_traindata - 1, 1) :
            normw += w[j] ** 2
        normw = math.sqrt(normw)
        for i in range(0, no_of_rows_trainlabels, 1) :
            if i in trainlabels :
                dp = dotproduct(w, traindata[i])
                if (trainlabels.get(i) * dp < 1) :
                    error += 1 - trainlabels[i] * dp
        error += c * normw ** 2
        for j in range(0, no_of_cols_traindata, 1) :
            dellf.append(0)

        for i in range(0, no_of_rows_traindata, 1) :
            if i in trainlabels :
                dp = dotproduct(w, traindata[i])
                if (trainlabels.get(i) * dp < 1) :
                    for j in range(0, no_of_cols_traindata, 1) :
                        dellf[j] += trainlabels.get(i) * traindata[i][j]
        for j in range(0, no_of_cols_traindata, 1) :
            dellf[j] -= 2 * c * w[j]
        # print(dellf)
        if abs(temp - error) <= stop :
            break
        for j in range(0, no_of_cols_traindata, 1) :
            w[j] = w[j] + eta * dellf[j]

            # calculate w
            print('NOW COMPUTING REGULARIZED HINGE LOSS...','\n')
            print("w = ", w)
            print('w0 = ', normw)
            print('\n \n \n')
            return w, normw

###################
######Adaptive learning rate for Hinge Loss
###################
def hinge_loss_adaptive_learningrate(traindata, trainlabels) :
    # Initialize w
    w = []
    for j in range(0, no_of_cols_traindata, 1) :
        w.append(float(0.02 * random.uniform(0, 1) - 0.01))

    # dellf descent iteration
    eta = 0.001
    diff = 1
    error = 0
    stop = 1
    count = 0

    # compute dellf and error
    while (diff > stop) :
        dellf = []
        dellf.extend(0 for _ in range(no_of_cols_traindata))

        for i in range(0, no_of_cols_traindata, 1) :
            if (trainlabels[i] != None) :
                label = trainlabels[i]
                for rowdata in traindata :
                    y = dotproduct(w, rowdata)
                    new_label = []
                    for l in label :
                        x = l * y
                        new_label.append(x)

                    a = new_label
                for j in range(0, no_of_cols_traindata) :
                    if j in a :
                        new = []
                        length = len(trainlabels[i])
                        for k in range(length) :
                            asd = -(trainlabels[i][k] * traindata[i][j])
                            new.append(asd)
                        dellf[j] = new

                    else :
                        dellf[j] = 0

        # print 'dellf: ', dellf
        # adaptive hinge
        eta_list = [1, .1, .01, .001, .0001, .00001, .000001,
                    .0000001, .00000001, .000000001,
                    .0000000001, .00000000001]
        bestobj = 1000000000000

        for k in range(0, len(eta_list), 1) :
            eta = eta_list[k]

            # update w
            for j in range(0, no_of_cols_traindata, 1) :
                for f in dellf :
                    w[j] -= eta * f

            # calculate error
            error = 0
            for m in range(0, no_of_cols_traindata) :
                if (trainlabels[m] != None) :
                    for t in trainlabels[m] :
                        error += max(0, 1 - t * dotproduct(w, traindata[m]))

            obj = error
            if obj < bestobj :
                bestobj = obj
                best_eta = eta

            # update w
        for j in range(0, no_of_cols_trainlabels, 1) :
            w[j] -= eta * dellf[j]
        #print('w: ', w[j])

        prevObj = error
        error = 0
        # compute error

        for n in range(0, no_of_rows_traindata, 1) :
            if (trainlabels[n] != None) :
                error += max(0, 1 - (n) * dotproduct(w, traindata[n]))
                break
        diff = abs(prevObj - error)

    # print 'diff:  ', diff
    # print("error: ", error)
    # print(best_eta)
    print('NOW COMPUTING HINGE LOSS ADAPTIVE LEARNING RATE...','\n')

    print ('w = ',  w[0:2])
    # distance from origin calculation
    normw = 0
    for j in range(0, no_of_cols_traindata - 1, 1) :
        normw += w[j] ** 2
    print('w0 = ', w[2])
    print('\n \n \n')
    return w, normw

#### MAIN #########

###################
#### Code to read train data and train labels
###################
train_data_file = sys.argv[1]
f = open(train_data_file)
traindata = []
i = 0
l = f.readline()
while (l != '') :
    a = l.split()
    l2 = []
    for j in range(1, len(a), 1) :
        l2.append(float(a[j]))
    traindata.append(l2)
    l = f.readline()

no_of_rows_traindata = len(traindata)
no_of_cols_traindata = len(traindata[0])
f.close()

train_label_file = sys.argv[1]
f = open(train_label_file)
trainlabels = []
i = 0
l = f.readline()

while (l != '') :
    a = l.split()
    l3 = []
    for j in range(0, 1, 1) :
        l3.append(int(a[j]))
    trainlabels.append(l3)
    l = f.readline()

# print(trainlabels)

no_of_rows_trainlabels = len(trainlabels)
no_of_cols_trainlabels = len(trainlabels[0])
f.close()

test_data_file = sys.argv[2]
f = open(test_data_file)
testdata = []
i = 0
l = f.readline()

while (l != '') :
    a = l.split()
    l4 = []
    for j in range(1, len(a), 1) :
        l4.append(float(a[j]))

    l4.append(1)
    testdata.append(l4)
    l = f.readline()
f.close()

test_label_file = sys.argv[2]
f = open(test_label_file)
testlabels = []
i = 0
l = f.readline()

while (l != '') :
    a = l.split()
    l5 = []
    for j in range(0, 1, 1) :
        l5.append(float(a[j]))
    testlabels.append(l5)
    l = f.readline()

no_of_rows_testlabels = len(testlabels)
no_of_cols_testlabels = len(testlabels[0])

f.close()

traindata, testdata = standardize_data(traindata, testdata)

w,w0 = least_squares(traindata, trainlabels)
w,w0 = least_squares_regularized(traindata, trainlabels)
w,w0 = hinge_loss(traindata, trainlabels)
w,w0 = hinge_loss_regularized(traindata, trainlabels)
w,w0 = hinge_loss_adaptive_learningrate(traindata, trainlabels)

print('THANK YOU')

