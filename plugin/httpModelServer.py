#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import urllib
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import cgi
import datetime

import numpy as np
import tensorflow as tf

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow import keras
from sys import argv
import sys


save_semantic_tokenizer = '../file/semantic_tokenizer'
f = open(save_semantic_tokenizer, 'r')
semantic_tokenizer = keras.preprocessing.text.tokenizer_from_json(f.read())
save_syntatic_tokenizer = '../file/syntatic_tokenizer'
f = open(save_syntatic_tokenizer, 'r')
syntatic_tokenizer = keras.preprocessing.text.tokenizer_from_json(f.read())
model_together_final=tf.keras.models.load_model(r'../file/model_together_final.h5')

num_word = 50000  # 词典的词数
voo_token = 'NONE'  # 指代缺失词
semantic_max_len = 300  # 句子最大长度
syntatic_max_len = 500  # 句子最大长度
trunc_type = 'post'
special_key={'[': '',']': ''}

class TodoHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # 获取参数
        params=None
        if '?' in self.path:  # 如果带有参数
            self.queryString = urllib.parse.unquote(self.path.split('?', 1)[1])
            params = urllib.parse.parse_qs(self.queryString)
        print(params)
        if params is not  None:
            if 'semantic' in params and 'syntatic' not in params:
                semantic=params['semantic'][0]
                for (key,value) in special_key.items():
                    semantic=semantic.replace(key,value)

                semantic = semantic.split(',')
                semantic = list(map(str, semantic))
                # 调用模型
                semantic_method_sequence = semantic_tokenizer.texts_to_sequences([semantic])
                semantic_method_padded = pad_sequences(semantic_method_sequence, maxlen=semantic_max_len,
                                                       padding=trunc_type, truncating=trunc_type)
                ret_message=semantic_method_padded
            elif 'syntatic' in params and 'semantic' not in params:
                syntatic=params['syntatic'][0]

                syntatic = syntatic.replace('[', '').replace(']', '')
                syntatic = syntatic.split(',')
                syntatic = list(map(str, syntatic))


                syntatic_method_sequence = syntatic_tokenizer.texts_to_sequences([syntatic])
                syntatic_method_padded = pad_sequences(syntatic_method_sequence, maxlen=syntatic_max_len,
                                                       padding=trunc_type, truncating=trunc_type)
                ret_message=syntatic_method_padded
            elif 'syntatic' in params and 'semantic' in params:
                semantic = params['semantic'][0]
                semantic = semantic.replace('[', '').replace(']', '')[:-1]
                semantic = semantic.split(',')
                semantic = list(map(float, semantic))

                syntatic = params['syntatic'][0]
                syntatic = syntatic.replace('[', '').replace(']', '')[:-1]
                syntatic = syntatic.split(',')
                syntatic = list(map(float, syntatic))


                semantic_vec_length_chain = 300 * 10
                syntatic_vec_length_chain = 500 * 10
                one_leaf_vec_padding_tmp = pad_sequences([syntatic], maxlen=syntatic_vec_length_chain,
                                                         padding="post", truncating="post", dtype='float32')
                syntatic = one_leaf_vec_padding_tmp[0]

                one_leaf_vec_padding_tmp = pad_sequences([semantic], maxlen=semantic_vec_length_chain,
                                                         padding="post", truncating="post", dtype='float32')
                semantic = one_leaf_vec_padding_tmp[0]

                # 预测单个数据
                syntatic = tf.reshape(syntatic, (1, 5000))
                semantic = tf.reshape(semantic, (1, 3000))
                result=model_together_final.predict([np.array(syntatic),np.array(semantic)])

                ret_message=result[0][0]
            else:
                ret_message = '参数错误，请联系管理员'
        else:
            ret_message = '参数错误，请联系管理员'
        # 返回结果
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(str(ret_message)).encode())





if __name__ == '__main__':
    host = ('127.0.0.1', 8003)
    server = HTTPServer(host, TodoHandler)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()