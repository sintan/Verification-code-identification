import tensorflow as tf
import captcha_model
import string
import numpy as np
from captcha.image import ImageCaptcha
import random
import os
from matplotlib import pyplot as plt
import cv2
from PIL import Image,ImageEnhance,ImageFilter


class train():
    def __init__(self,
                 width=82,  # 验证码图片的宽
                 height=32,  # 验证码图片的高
                 char_num=4,  # 验证码字符个数
                 characters=string.digits + string.ascii_uppercase + string.ascii_lowercase,
                 rootdir="./train_img"):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)
        self.rootdir = rootdir

    def get_img(self, batch_size=50):
        X = np.zeros([batch_size, self.height, self.width, 1])
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        Y = np.zeros([batch_size, self.char_num, self.classes])
        image = ImageCaptcha(width=self.width, height=self.height)

        while True:
            for i in range(batch_size):
                rootdir = self.rootdir
                file_names = []
                for parent, dirnames, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
                    file_names = filenames
                x = random.randint(0, len(file_names) - 1)
                file = str(file_names[x])
                strn = file[0:4]
                captcha_str = ''.join(strn)
                img = Image.open(rootdir+ '/' + file).convert('L')
                img = np.array(img.getdata())
                X[i] = np.reshape(img, [self.height, self.width, 1]) / 255.0
                for j, ch in enumerate(captcha_str):
                    Y[i, j, self.characters.find(ch)] = 1
            Y = np.reshape(Y, (batch_size, self.char_num * self.classes))
            yield X, Y

    def train_main(self, filename='./model/model-1.ckpt'):
        x = tf.placeholder(tf.float32, [None, self.height, self.width, 1])
        y_ = tf.placeholder(tf.float32, [None, self.char_num * self.classes])
        keep_prob = tf.placeholder(tf.float32)

        model = captcha_model.captchaModel(self.width, self.height, self.char_num, self.classes)
        y_conv = model.create_model(x, keep_prob)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        predict = tf.reshape(y_conv, [-1, self.char_num, self.classes])
        real = tf.reshape(y_, [-1, self.char_num, self.classes])
        correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y = next(self.get_img(128))
                _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
                print('step:%d,loss:%f' % (step, loss))
                if step % 100 == 0:
                    batch_x_test, batch_y_test = next(self.get_img(512))
                    acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                    print('###############################################step:%d,accuracy:%f' % (step, acc))
                    if acc > 0.99:
                        # 保存模型的名字，名字同样会进行覆盖
                        saver.save(sess, filename)
                        break
                step += 1
        print('train success')

    def extrain(self, filename = './model/model-1.ckpt',restore_file = './capcha_model-5.ckpt' ):
        x = tf.placeholder(tf.float32, [None, self.height, self.width, 1])
        y_ = tf.placeholder(tf.float32, [None, self.char_num * self.classes])
        keep_prob = tf.placeholder(tf.float32)

        model = captcha_model.captchaModel(self.width, self.height, self.char_num, self.classes)
        y_conv = model.create_model(x, keep_prob)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        predict = tf.reshape(y_conv, [-1, self.char_num, self.classes])
        real = tf.reshape(y_, [-1, self.char_num, self.classes])
        correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, restore_file)
            step = 0
            while True:
                batch_x, batch_y = next(self.get_img(128))
                _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
                print('step:%d,loss:%f' % (step, loss))
                if step % 100 == 0:
                    batch_x_test, batch_y_test = next(self.get_img(512))
                    acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                    print('###############################################step:%d,accuracy:%f' % (step, acc))
                    if acc > 0.99:
                        # 保存模型的名字，名字同样会进行覆盖
                        saver.save(sess, filename)
                        break
                step += 1
        print('train success')




    def get_parameter(self):
        return self.width, self.height, self.char_num, self.characters, self.classes

    def pre_treat(sellf,filename):
        # 去除干扰线
        im = Image.open(filename)
        # 把输入的验证码图片转成82*32大小
        im = im.resize((82, 32), Image.ANTIALIAS)
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        # 图像转为'RGB'模式
        im = im.convert('RGB')
        data = im.getdata()
        w, h = im.size
        # im.show()
        black_point = 0
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                r, g, b = im.getpixel((x, y))
                # 此处去除颜色的范围是试出来的，留下了深红色
                if (r < g + b):
                    r = 255
                    g = 255
                    b = 255
                    im.putpixel((x, y), (r, g, b))
        # 图像二值化
        im = im.convert('L')
        return im

    def prewipe(self,target='./train_img',rootdir = './zs_img'):
        for parent, dirnames, filenames in os.walk(rootdir):
            file_names = filenames
        for x in file_names:
            img = self.pre_treat(rootdir + '/' + x)
            img.save(target + x[0:4] + '.png')


if __name__ == '__main__':
    train1 = train()
    # train1.train_main('./model-1.ckpt')
    # train1.extrain()