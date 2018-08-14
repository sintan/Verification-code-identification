from PIL import Image
import tensorflow as tf
import numpy as np
import captcha_model
import os
import operator as op
import train_1
import string

class predict():
    def __init__(self,
                 width = 82,#验证码图片的宽
                 height = 32,#验证码图片的高
                 char_num = 4,#验证码字符个数
                 characters=string.digits + string.ascii_uppercase + string.ascii_lowercase,
                 modeldir="./capcha_model-5.ckpt"):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)
        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        model = captcha_model.captchaModel(self.width, self.height, self.char_num, self.classes)
        y_conv = model.create_model(self.x, self.keep_prob)
        self.predict = tf.argmax(tf.reshape(y_conv, [-1, self.char_num, self.classes]), 2)
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        # 使用的模型
        self.modeldir = modeldir
        self.saver.restore(self.sess, self.modeldir)

    def file_exist(self,file):
        result = os.path.exists(file)
        return result

    def file_type(self,file):
        result = os.path.exists(file+'/')
        return result

    def single_file(self,filename):
        sin_result = 'null'
        train1 = train_1.train()
        if os.path.exists(filename):
            gray_image = train1.pre_treat(filename)
            img = np.array(gray_image.getdata())
            test_x = np.reshape(img, [self.height, self.width, 1]) / 255.0
            pre_list = self.sess.run(self.predict, feed_dict={self.x: [test_x], self.keep_prob: 1})
            for i in pre_list:
                sin_result = ''
                for j in i:
                    sin_result += self.characters[j]
                # print(sin_result)

        return sin_result


    def multi_file(self, rootdir, print_accurate=True):
        xn = 0
        # right为识别正确的验证码个数，如果要看正确率可以本函数的注释取消
        right=0
        train1 = train_1.train()
        file_names = []
        for parent, dirnames, filenames in os.walk(rootdir):
            file_names = filenames
        mul_result = []
        while xn < len(file_names) - 1:
            filename = file_names[xn]
            gray_image = train1.pre_treat(rootdir + '/' + filename)
            img = np.array(gray_image.getdata())
            test_x = np.reshape(img, [self.height, self.width, 1]) / 255.0
            pre_list = self.sess.run(self.predict, feed_dict={self.x: [test_x], self.keep_prob: 1})
            s = ''
            for i in pre_list:
                s = ''
                for j in i:
                    s += self.characters[j]
                # 显示识别的验证码和文件名包含的正确验证码
                # print(s)
                # print(file_names[xn][0:4])
            mul_result.append(s)
            if op.eq(s,file_names[xn][0:4]):
                right += 1
            xn += 1
        if print_accurate:
            print('准确率' + str(right / len(file_names) * 100) + '%')
        return mul_result

    def file_predict(self,rootdir):
        rlist=[]
        if type(rootdir)== list:
            for i in rootdir:
                if self.file_exist(i):
                    if self.file_type(i):
                        rlist.append(self.multi_file(i))
                    else:
                        rlist.append(self.single_file(i))
                else:
                    rlist.append('not exist')
        elif type(rootdir)== str:
            if self.file_exist(rootdir):
                if self.file_type(rootdir):
                    rlist.append(self.multi_file(rootdir))
                else:
                    rlist.append(self.single_file(rootdir))
            else:
                rlist.append('not exist')
        else:
            rlist.append('type error')
        return rlist

if __name__ == '__main__':
    test = predict()
    print(test.file_predict('./test1/4pkx_TmAGeWV1.png'))
    print(test.file_predict('./test'))
    print(test.file_predict('./test1'))

