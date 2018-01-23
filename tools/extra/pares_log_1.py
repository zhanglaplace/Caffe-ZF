import numpy as np
import os
import sys
import matplotlib.pyplot as plt



#os.system('cat ')

def plotCurse(logfile):
    command_train_loss = 'cat ' + logfile + ' | grep \'Train net output #0\' | awk \'{print $11}\''
    command_train_loss_1 = 'cat ' + logfile + ' | grep \'Train net output #1\' | awk \'{print $11}\''
    command_train_iter = 'cat ' + logfile + ' | grep \'218] Iteration\' | awk \'{print $6}\''
    command_test_loss = 'cat ' + logfile + ' | grep \'Test net output #0\' | awk \'{print $11}\''
    command_test_loss_1 = 'cat ' + logfile + ' | grep \'Test net output #2\' | awk \'{print $11}\''
    command_test_accuracy = 'cat ' + logfile + ' | grep \'Test net output #1\' | awk \'{print $11}\''
    command_test_iter = 'cat ' + logfile + ' | grep \'330] Iteration\' | awk \'{print $6}\''
    train_loss = os.popen(command_train_loss).readlines()
    train_loss_center = os.popen(command_train_loss_1).readlines()
    test_loss = os.popen(command_test_loss).readlines()
    test_loss_center = os.popen(command_test_loss_1).readlines()
    test_accuracy = os.popen(command_test_accuracy).readlines()
    train_iter = os.popen(command_train_iter).readlines()
    test_iter = os.popen(command_test_iter).readlines()
    np_train_loss = [round(float(line.strip()), 5) for line in train_loss]
    np_train_loss_center = [round(float(line.strip()), 5) for line in train_loss_center]
    np_train_iter = [line.strip() for line in train_iter]
    np_test_loss = [round(float(line.strip()), 5) for line in test_loss]
    np_test_loss_center = [round(float(line.strip()), 5) for line in test_loss_center]
    np_test_iter = [line.strip() for line in test_iter]
    np_test_accuracy = [round(float(line.strip()), 5) for line in test_accuracy]
    # for i in range(len(np_test_iter)):
    # print(np_test_iter[i])
    # print (len(np_test_loss))
    # print(len(np_test_accuracy))
    plt.figure(1)

    np_test_iter = map(eval, np_test_iter)
    np_train_iter = map(eval, np_train_iter)
    plt.figure(1)
    plt.subplot(221)
    plt.xlabel('iterations')
    plt.ylabel('test_loss')
    plt.title('test loss curse')
    plt.plot(np_test_iter, np_test_loss, 'b', label='softmax_loss', linewidth=2)  # test loss
    plt.hold
    plt.plot(np_test_iter, np_test_loss_center, 'r', label='center_loss', linewidth=2)  # test loss
    plt.legend()

    plt.subplot(222)
    plt.xlabel('iterations')
    plt.ylabel('test_accuracy')
    plt.title('test accuracy curse')
    plt.plot(np_test_iter, np_test_accuracy, "b", label="request delay")  # test accuracy

    plt.subplot(212)
    plt.plot(np_train_iter, np_train_loss_center, "b-", label="train_center_loss")  # test accuracy
    plt.hold
    plt.plot(np_train_iter, np_train_loss, 'r', label="train_softmax_loss")  # train loss
    plt.xlabel('iterations')
    plt.ylabel('train_loss')
    plt.title('train_loss curse')
    plt.legend()
    plt.show()


def drawSphereface(logfile):
    command_lambda = 'cat ' + logfile + ' | grep \'Train net output #0\' | awk \'{print $11}\''
    command_train_loss = 'cat ' + logfile + ' | grep \'Train net output #1\' | awk \'{print $11}\''
    command_train_iter = 'cat ' + logfile + ' | grep \'218] Iteration\' | awk \'{print $6}\''
    train_loss = os.popen(command_train_loss).readlines()
    train_lamdba = os.popen(command_lambda).readlines()
    train_iter = os.popen(command_train_iter).readlines()
    np_train_loss = [round(float(line.strip()), 5) for line in train_loss]
    np_train_lamdba = [round(float(line.strip()), 5) for line in train_lamdba]
    np_train_iter = [line.strip() for line in train_iter]

    plt.figure(1)
    np_train_iter = map(eval, np_train_iter)
    plt.plot(np_train_iter, np_train_lamdba, "b-", label="train_lambda")  # test accuracy
    plt.xlabel('iterations')
    plt.ylabel('lambda')
    plt.title('lambda')
    plt.figure(2)
    plt.plot(np_train_iter, np_train_loss, 'r', label="train_loss")  # train loss
    plt.xlabel('iterations')
    plt.ylabel('train_loss')
    plt.title('train_loss curse')
    plt.legend()
    plt.show()

def NormScale(feature_file,dst_file,alpha):
    fid = open(feature_file,'r')
    fs = open(dst_file,'w')
    files = fid.readlines()
    for file in files:
        data = file.split()
        sum = 0
        fs.write(data[0]+'\t')
        for i in range(1,3):
            sum += (float)(data[i])*(float)(data[i])
        for i in range(1, 3):
            data[i] = alpha*((float)(data[i])/(np.sqrt(sum)+1e-5))
        fs.write(str(data[1])+'\t')
        fs.write(str(data[2])+'\n')
    fs.close()
    fid.close()

if __name__ == '__main__':
    #logfile = 'INFO2017-12-18T14-45-16.txt20171218-144516.12325'
    feature_file = sys.argv[1]#'/home/zf/deeplearning/caffe/face_examples/face_recognition/studyFine/feat_center_train_1.txt'
    #dst_file =sys.argv[2]#'/home/zf/deeplearning/caffe/face_examples/face_recognition/studyFine/feat_center_train_1_norm.txt'
    #NormScale(feature_file,dst_file,5)
    drawSphereface(feature_file)


