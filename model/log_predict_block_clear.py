# 作者：常晓松
# 作用：日志位置预测
# 时间： 2022/3/23 14:35
import io
import sys
import os
import io
import sys
import os
from sys import argv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pymysql
from keras.layers import Convolution1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import time
import pickle
import operator
from itertools import count
import random


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_value(y_true, y_pred):
    import sklearn as sk
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 5))

    plt.cla()

    from sklearn.metrics import balanced_accuracy_score
    balance_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    f1_value = sk.metrics.f1_score(y_true, y_pred)

    x = ['balance_accuracy', 'precision', 'recall', 'f1_value']
    y_myself = [balance_accuracy, precision, recall, f1_value]
    y_reference_Within = [0.801, 0.506, 0.618, 0.551]

    y_reference_Cross = [0.673, 0.354, 0.435, 0.386]

    # 保留两位小数
    labels = x
    y_myself = y_myself
    name_my_model = 'My model'
    y_reference_Within = y_reference_Within
    name_reference_Within = 'reference_within'
    y_reference_Cross = y_reference_Cross
    name_reference_Cross = 'reference_cross'

    y_myself_tmp = []
    for one in y_myself:
        y_myself_tmp.append(round(one, 4))
    y_myself = y_myself_tmp
    y_reference_tmp = []
    for one in y_reference_Within:
        y_reference_tmp.append(round(one, 4))
    y_reference_Within = y_reference_tmp
    y_reference_Cross_tmp = []
    for one in y_reference_Cross:
        y_reference_Cross_tmp.append(round(one, 4))
    y_reference_Cross = y_reference_Cross_tmp

    plt.rcParams['font.sans-serif'] = ['SimHei']
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars
    x = x * 2
    ax = plt.gca()
    rects1 = ax.bar(x - width, y_myself, width, label=name_my_model)
    rects2 = ax.bar(x, y_reference_Within, width, label=name_reference_Within)
    rects3 = ax.bar(x + width, y_reference_Cross, width, label=name_reference_Cross)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.5)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    save_pic(plt, "多指标")
    # plt.show()
    return balance_accuracy, precision, recall, f1_value

def float_revert_int(value_list):
    revert_value_list = []
    for i in value_list:
        revert_value_list.append(int(np.round(i)))
    value_list = revert_value_list
    return revert_value_list
def draw_confusion_matrix(y_true, y_pred, dic_lables):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    plt.cla()

    labels = []
    for key in dic_lables:
        labels.append(key)
    print_message(type(labels))
    # 小数转整数
    y_pred_int = float_revert_int(y_pred)
    sns.set()
    f, ax = plt.subplots()
    C2 = confusion_matrix(y_true,y_pred,  labels=labels)
    print_message(C2)  # 打印出来看看
    sns.heatmap(C2, annot=True, fmt='.20g', ax=ax, cmap="YlGnBu")  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('true')  # x轴
    ax.set_ylabel('predict')  # y轴
    save_pic(plt, '混淆矩阵')
    # plt.show()
    return plt

def save_pic(plt, file_name):
    # 创建目录
    import os, time
    dirs = 'C:\\Users\\chang\\Desktop\\日志工作空间\\实验图片\\'
    t = time.strftime('%Y-%m-%d-%H', time.localtime(int(time.time())))
    dirs = dirs + t
    file = dirs + '\\' + file_name + '.png'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if os.path.exists(file):
        os.remove(file)
    # 保存图片
    plt.savefig(file)  # 保存图片

def plot_graphs(history_loc, type_loc):
    import matplotlib.pyplot as plt
    plt.cla()

    plt.plot(history_loc.history[type_loc])
    plt.plot(history_loc.history['val_' + type_loc])
    plt.xlabel('Epochs')
    plt.ylabel(type_loc)
    plt.legend([type_loc, 'val_' + type_loc])
    save_pic(plt, type_loc)
    # plt.show()
    return plt
def get_call_graph_model_together():
    model1 = tf.keras.models.Sequential([
        keras.layers.InputLayer(call_graph_struct_vec_length_chain)
    ])
    model2 = tf.keras.models.Sequential([
        (Convolution1D(input_shape=(call_graph_semantic_vec_length_chain, 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
    ])
    model3 = tf.keras.models.Sequential([
        (Convolution1D(input_shape=(call_graph_syntatic_vec_length_chain, 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
    ])
    model_together = keras.layers.concatenate([model3.output, model2.output])
    x = Dropout(0.4)(model_together)
    x = Dense(24, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    final_output = Dense(24, activation='relu')(x)
    model_together = Model(inputs=[model3.input, model2.input], outputs=final_output)
    return model_together

def seperate_data():
    global struct_train_vec, struct_test_vec, \
        semantic_train_vec, semantic_test_vec, \
        syntatic_train_vec, syntatic_test_vec, \
        call_graph_struct_train_vec, call_graph_struct_test_vec, \
        call_graph_semantic_train_vec, call_graph_semantic_test_vec, \
        call_graph_syntatic_train_vec, call_graph_syntatic_test_vec, \
        call_graph_num_train_vec,call_graph_num_test_vec,\
        struct_train_labels, struct_test_labels, \
        struct_verify_vec, semantic_verify_vec, \
        syntatic_verify_vec, call_graph_semantic_verify_vec,call_graph_num_verify_vec,call_graph_struct_verify_vec, \
        call_graph_syntatic_verify_vec, struct_verify_labels, struct_verify_leaf
    from sklearn.model_selection import train_test_split  # 出留法
    struct_train_vec, struct_test_vec, \
    semantic_train_vec, semantic_test_vec, \
    syntatic_train_vec, syntatic_test_vec, \
    call_graph_struct_train_vec, call_graph_struct_test_vec, \
    call_graph_semantic_train_vec, call_graph_semantic_test_vec, \
    call_graph_syntatic_train_vec, call_graph_syntatic_test_vec, \
    call_graph_num_train_vec, call_graph_num_test_vec, \
    struct_train_labels, struct_test_labels, \
    struct_train_leaf, struct_test_leaf = train_test_split(
        struct_leaf_vec, semantic_leaf_vec, syntatic_leaf_vec,
        call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec,
        call_graph_num_vec,
        lable, leaf,
        test_size=0.4,
        random_state=seed)
    struct_verify_vec, struct_test_vec, \
    semantic_verify_vec, semantic_test_vec, \
    syntatic_verify_vec, syntatic_test_vec, \
    call_graph_struct_verify_vec, call_graph_struct_test_vec, \
    call_graph_semantic_verify_vec, call_graph_semantic_test_vec, \
    call_graph_syntatic_verify_vec, call_graph_syntatic_test_vec, \
    call_graph_num_verify_vec, call_graph_num_test_vec, \
    struct_verify_labels, struct_test_labels, \
    struct_verify_leaf, struct_test_leaf = train_test_split(
        struct_test_vec, semantic_test_vec, syntatic_test_vec,
        call_graph_struct_test_vec, call_graph_semantic_test_vec, call_graph_syntatic_test_vec,call_graph_num_test_vec,

        struct_test_labels,
        struct_test_leaf,
        test_size=0.5,
        random_state=seed)

def shuffle_data(struct_vec, semantic_vec, syntatic_vec,
                 call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec,
                 leaf, labels,call_graph_num_vec, seed):
    import random
    c = list(zip(struct_vec, semantic_vec, syntatic_vec,
                 call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec,
                 leaf, labels,call_graph_num_vec))  # 将a,b整体作为一个zip,每个元素一一对应后打乱
    random.seed(seed)
    random.shuffle(c)  # 打乱c
    struct_vec[:], semantic_vec[:], syntatic_vec[:], \
    call_graph_struct_leaf_vec[:], call_graph_semantic_leaf_vec[:], call_graph_syntatic_leaf_vec[:], \
    leaf[:], labels[:],call_graph_num_vec[:] = zip(*c)  # 将打乱的c解开
    return struct_vec, semantic_vec, syntatic_vec, call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec, leaf, labels,call_graph_num_vec


def get_chair_vec(vec_chair, i):
    one_leaf_list = []
    leaf = True
    message_type = i
    for one_leaf in vec_chair:
        sqli = "select a.vectorSemantic,REPLACE(REPLACE(vectorStruct,'[',''),']',''),logNum ,REPLACE(REPLACE(syntacticMessage,'[',''),']','') " \
               "from data_model_2 a  where seq=" + str(one_leaf) + ";"
        result = cur.execute(sqli)
        oneRow = cur.fetchone()
        vectorSemantic = oneRow[0]
        vectorStruct = oneRow[1]
        logNum = oneRow[2]
        syntacticMessage = oneRow[3]
        if vectorSemantic != "[]":
            if message_type == 1:
                if vectorStruct is not None:
                    list_struct = vectorStruct.split(',')
                    list_struct = list(map(float, list_struct))
                    one_leaf_list.extend(list_struct)
                    # one_leaf_list.extend([0,0,0,0])
            if message_type == 2:
                if vectorSemantic is not None and len(vectorSemantic) > 0:
                    list_semantic = vectorSemantic.split(',')
                    list_semantic = list(map(float, list_semantic))
                    one_leaf_list.extend(list_semantic)
                    # one_leaf_list.extend([0,0,0,0])
            if message_type == 3:
                if syntacticMessage is not None and len(syntacticMessage) > 0:
                    list_syntactic = syntacticMessage.split(',')
                    list_syntactic = list(map(float, list_syntactic))
                    one_leaf_list.extend(list_syntactic)
                    # one_leaf_list.extend([0,0,0,0])
    # one_leaf_list.extend([0,0,0,0])
    return one_leaf_list

def get_parent(vec_chair):
    cur = conn.cursor()
    sqli = "select parentId from data_model_2 where seq=" + str(vec_chair[len(vec_chair) - 1]) + ";"
    result = cur.execute(sqli)
    oneRow = cur.fetchone()
    if oneRow[0] == 0:
        return vec_chair
    else:
        vec_chair.append(oneRow[0])
        return get_parent(vec_chair)

def print_message(message):
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    print('时间：' + t, end=" ", flush=True)
    print(message, flush=True)

def get_tree_model_together():
    model1 = tf.keras.models.Sequential([
        keras.layers.InputLayer(struct_vec_length_chain)
    ])
    model2 = tf.keras.models.Sequential([
        (Convolution1D(input_shape=(semantic_vec_length_chain, 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
    ])
    model3 = tf.keras.models.Sequential([
        (Convolution1D(input_shape=(syntatic_vec_length_chain, 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
    ])
    model_together = keras.layers.concatenate([model3.output, model2.output])
    x = Dropout(0.4)(model_together)
    x = Dense(24, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    final_output = Dense(24, activation='relu')(x)
    model_together = Model(inputs=[model3.input, model2.input], outputs=final_output)
    return model_together

def get_connection():
    global conn
    host = "127.0.0.1"
    user = "root"
    password = ''
    db = 'predict_log'
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset='utf8',
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    return conn



if __name__ == '__main__':


    project_name = 'druid-master'
    # 1-语法 2-语义 3-结合(本树+调用关系)
    # 4-语法（新）5-纯调用关系 6-结合(本树)
    select_model = 6
    if len(argv) >= 3:
        project_name = argv[1]
        select_model = int(argv[2])
    print(project_name,select_model)
    struct_vec_length_chain = 56 * 10
    semantic_vec_length_chain = 300 * 10
    syntatic_vec_length_chain = 500 * 10

    # 层次图 新增的向量维度
    call_graph_struct_vec_length_chain = 56 * 10
    call_graph_semantic_vec_length_chain = 300 * 10
    call_graph_syntatic_vec_length_chain = 500 * 10

    bro_vec_length_semantic = 0  # 304
    vec_length = semantic_vec_length_chain + bro_vec_length_semantic
    seed =  1234 #random.randint(1, 1000)
    # 获取每个叶子节点
    print_message("获取每个叶子节点")
    leaf = []
    lable = []
    struct_leaf_vec = []
    semantic_leaf_vec = []
    syntatic_leaf_vec = []

    call_graph_struct_leaf_vec = []
    call_graph_semantic_leaf_vec = []
    call_graph_syntatic_leaf_vec = []
    call_graph_num_vec = []

    conn = get_connection()
    cur = conn.cursor()
    sqli = "select seq,CASE WHEN logNum > 0 THEN  1  ELSE 0 END " \
           "from  data_model_2 a where   logNum<>0 and leaf='Y' " \
           " and  methodSeq in (select seq from data_model_1 where ProjectName = '"+project_name+"') " \
           "    "
    result = cur.execute(sqli)
    for i in range(result):
        oneRow = cur.fetchone()
        leaf.append(oneRow[0])
        lable.append(oneRow[1])


    # 随机采样得到负样本
    cur = conn.cursor()
    negative_seq_list = []
    sqli = "select seq  from data_model_2 a where   logNum=0" \
           " and leaf='Y'  " \
           " and  methodSeq in (select seq from data_model_1 where ProjectName = '"+project_name+"') "

    result = cur.execute(sqli)
    for i in range(result):
        oneRow = cur.fetchone()
        negative_seq_list.append(oneRow[0])

    random.seed(seed)
    negative_seq_list = random.sample(negative_seq_list, len(leaf))

    for i in negative_seq_list:
        cur = conn.cursor()
        sqli = "select seq,CASE WHEN logNum > 0 THEN  1  ELSE 0 END from data_model_2 where seq= " + str(i) + ";"
        result = cur.execute(sqli)
        for i in range(result):
            oneRow = cur.fetchone()
            leaf.append(oneRow[0])
            lable.append(oneRow[1])
    print_message("获取叶子节点的所有父节点")
    for one_leaf in leaf:
        vec_chain = []
        # 叶子节点放入 list 中
        vec_chain.append(one_leaf)
        # 获取叶子节点的所有父节点 放入List中
        vec_chain = get_parent(vec_chain)

        vec_chain_call_graph = []
        cur_2 = conn.cursor()
        sql = "select seq from data_model_2 " \
              "where methodSeq in " \
              "(select calledMethodSeq " \
              "from call_graph_data where callBlockSeq=" + str(vec_chain[0]) + ") " \
                                                                               "and leaf='Y';"
        result_2 = cur_2.execute(sql)
        for i in range(result_2):
            one_row = cur_2.fetchone()
            vec_chain_call_graph.append(one_row[0])

        for i in range(1, 4):
            call_graph_maxlen = 0
            if i == 1:
                maxlen = struct_vec_length_chain
                call_graph_maxlen = call_graph_struct_vec_length_chain

            elif i == 2:
                maxlen = semantic_vec_length_chain
                call_graph_maxlen = call_graph_semantic_vec_length_chain

            else:
                maxlen = syntatic_vec_length_chain
                call_graph_maxlen = call_graph_syntatic_vec_length_chain

            # 根据list中的记录查询节点记录，组合向量
            call_graph_vec=[]
            one_leaf_vec = get_chair_vec(vec_chain, i)
            # 判断是否存在调用信息，若存在则拼接
            call_graph_num = len(vec_chain_call_graph)
            if len(vec_chain_call_graph) > 0:
                call_graph_vec = get_chair_vec(vec_chain_call_graph, i)

            # 划分为等长向量
            one_leaf_vec_padding_tmp = pad_sequences([one_leaf_vec], maxlen=maxlen,
                                                     padding="post", truncating="post", dtype='float32')
            one_leaf_vec_padding_tmp = one_leaf_vec_padding_tmp[0]

            # 上级树向量
            call_graph_num_vec.append(call_graph_num)
            if call_graph_maxlen > 0:
                call_graph_vec_padding_tmp = pad_sequences([call_graph_vec], maxlen=call_graph_maxlen,
                                                           padding="post", truncating="post", dtype='float32')
                call_graph_vec_padding_tmp = call_graph_vec_padding_tmp[0]

                if i == 1:
                    call_graph_struct_leaf_vec.append(call_graph_vec_padding_tmp)
                elif i == 2:
                    call_graph_semantic_leaf_vec.append(call_graph_vec_padding_tmp)
                else:
                    call_graph_syntatic_leaf_vec.append(call_graph_vec_padding_tmp)

            if i == 1:
                struct_leaf_vec.append(one_leaf_vec_padding_tmp)
            elif i == 2:
                semantic_leaf_vec.append(one_leaf_vec_padding_tmp)
            else:
                syntatic_leaf_vec.append(one_leaf_vec_padding_tmp)

    cur.close()
    conn.close()
    print_message("完成")

    # 打乱数据
    print_message('打乱数据')
    struct_leaf_vec, semantic_leaf_vec, syntatic_leaf_vec, \
    call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec, \
    leaf, lable,call_graph_num_vec = shuffle_data(struct_leaf_vec, semantic_leaf_vec, syntatic_leaf_vec,
                               call_graph_struct_leaf_vec, call_graph_semantic_leaf_vec, call_graph_syntatic_leaf_vec,
                               leaf, lable,call_graph_num_vec, seed)

    # 独热编码
    lable = tf.keras.utils.to_categorical(lable, num_classes=2)

    # 划分数据集
    seperate_data()

    # 测试需要 后续去除
    # semantic_train_vec = call_graph_semantic_train_vec
    # semantic_test_vec = call_graph_semantic_test_vec
    #

    #
    # syntatic_train_vec = call_graph_syntatic_train_vec
    # syntatic_test_vec = call_graph_syntatic_test_vec
    # syntatic_verify_vec = call_graph_syntatic_verify_vec

    print_message('完成')

    from keras import regularizers, Model

    print_message("组建模型")
    # 组建模型
    num_word = 50000  # 词典的词数
    embedding_dim = 48
    max_len = vec_length
    # 获取两个神经网络

    tree_model_together = get_tree_model_together()
    call_graph_model_together = get_call_graph_model_together()

    # 拼接两个 神经网络
    model_together_final = keras.layers.concatenate([tree_model_together.output, call_graph_model_together.output])
    x = Dense(128, activation='relu')(model_together_final)
    final_output = Dense(24, activation='relu')(x)

    model_together_final = Model(inputs=[tree_model_together.input, call_graph_model_together.input],
                                 outputs=final_output)

    model_call_graph_num_model = tf.keras.models.Sequential([
        keras.layers.InputLayer(1)
    ])
    model_together_final_tmp = keras.layers.concatenate([model_together_final.output, model_call_graph_num_model.output])
    x = Dense(128, activation='relu')(model_together_final_tmp)
    x = Dense(24, activation='relu')(x)
    final_output = Dense(2, activation='softmax')(x)


    model_together_final = Model(inputs=[model_together_final.input, model_call_graph_num_model.input],
                                 outputs=final_output)

    model_struct = tf.keras.models.Sequential([
        (keras.layers.InputLayer(struct_vec_length_chain )),
        (Dropout(0.4)),
        (Dense(24, activation='relu')),
        (Dense(128, activation='relu')),
        (Dense(24, activation='relu')),

        Dense(2, activation='softmax'),
    ])
    model_semantic = tf.keras.models.Sequential([

        (Convolution1D(input_shape=(semantic_vec_length_chain , 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
        (Dropout(0.4)),
        (Dense(24, activation='relu')),
        (Dense(128, activation='relu')),
        (Dense(24, activation='relu')),

        Dense(2, activation='softmax'),
    ])

    model_syntatic = tf.keras.models.Sequential([

        (Convolution1D(input_shape=(syntatic_vec_length_chain , 1), filters=1,
                       kernel_size=1, strides=1, padding='same',
                       activation='relu', kernel_initializer=keras.initializers.Ones(),
                       bias_initializer='zeros')),
        (MaxPooling1D(pool_size=2, strides=2, padding='same', )),
        (Flatten()),
        (keras.layers.Embedding(num_word, embedding_dim)),
        (keras.layers.GlobalMaxPool1D()),
        (Dropout(0.4)),
        (Dense(24, activation='relu')),
        (Dense(128, activation='relu')),
        (Dense(24, activation='relu')),

        Dense(2, activation='softmax'),
    ])

    if select_model == 1:
        model_together_final = model_struct
    elif select_model == 2:
        model_together_final = model_semantic
    elif select_model == 3:
        model_together_final = model_together_final
    elif select_model == 4:
        model_together_final = model_syntatic
    elif select_model == 5:
        model_call_graph_num_model = tf.keras.models.Sequential([
            keras.layers.InputLayer(1)
        ])
        model_tmp = keras.layers.concatenate([call_graph_model_together.output, model_call_graph_num_model.output])
        x=Dense(24, activation='relu')(model_tmp)
        x=Dense(128, activation='relu')(x)
        x=Dense(24, activation='relu')(x)
        final_output = Dense(2, activation='softmax')(x)
        model_together_final = Model(inputs=[call_graph_model_together.input,model_call_graph_num_model.input],
                                     outputs=final_output)
    elif select_model == 6:
        model_tmp = keras.layers.concatenate([tree_model_together.output])
        final_output = Dense(2, activation='softmax')(model_tmp)
        model_together_final = Model(inputs=[tree_model_together.input],
                                     outputs=final_output)
        model_together_final = model_together_final
    model_together_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_together_final.summary()
    # 训练

    num_epoch = 10  # 训练周期
    print_message("训练")
    # from keras.callbacks import ReduceLROnPlateau
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, mode='auto')

    # history =model1.fit(np.array(struct_train_vec), np.array(struct_train_labels), epochs=num_epoch,
    #                         validation_data=(np.array(struct_test_vec), np.array(struct_test_labels)),
    #                                          callbacks=[reduce_lr], verbose=2)

    history = []
    if select_model == 1:
        history = model_together_final.fit(np.array(struct_train_vec),
                                           np.array(struct_train_labels),
                                           validation_data=(np.array(struct_test_vec),
                                                            np.array(struct_test_labels)),
                                           epochs=num_epoch, verbose=2)
    elif select_model == 2:
        history = model_together_final.fit(np.array(semantic_train_vec),
                                           np.array(struct_train_labels),
                                           validation_data=(np.array(semantic_test_vec),
                                                            np.array(struct_test_labels)),
                                           epochs=num_epoch, verbose=2)
    elif select_model == 3:
        history = model_together_final \
            .fit([np.array(syntatic_train_vec), np.array(semantic_train_vec),
                  np.array(call_graph_syntatic_train_vec), np.array(call_graph_semantic_train_vec),np.array(call_graph_num_train_vec)],
                 np.array(struct_train_labels),
                 validation_data=([np.array(syntatic_test_vec),
                                   np.array(semantic_test_vec),
                                   np.array(call_graph_syntatic_test_vec),
                                   np.array(call_graph_semantic_test_vec),np.array(call_graph_num_test_vec)
                                   ],
                                  np.array(struct_test_labels)),
                 epochs=num_epoch, verbose=2)
    elif select_model == 4:
        history = model_together_final.fit(np.array(syntatic_train_vec),np.array(struct_train_labels),
                                           validation_data=(np.array(syntatic_test_vec),
                                                            np.array(struct_test_labels)),
                                           epochs=num_epoch, verbose=2)
    elif select_model == 5:
        history = model_together_final \
            .fit([np.array(call_graph_syntatic_train_vec),
                  np.array(call_graph_semantic_train_vec),
                  np.array(call_graph_num_train_vec)],
                 np.array(struct_train_labels),
                 validation_data=([np.array(call_graph_syntatic_test_vec),
                                      np.array(call_graph_semantic_test_vec),
                                   np.array(call_graph_num_test_vec)
                                  ],
                                  np.array(struct_test_labels)),
                 epochs=num_epoch, verbose=2)
    elif select_model == 6:
        history = model_together_final \
            .fit([np.array(syntatic_train_vec), np.array(semantic_train_vec)
                  ],
                 np.array(struct_train_labels),
                 validation_data=([np.array(syntatic_test_vec),
                                   np.array(semantic_test_vec)
                                   ],
                                  np.array(struct_test_labels)),
                 epochs=num_epoch, verbose=2)

    # 输出结果
    print_message("开始")

    predict_number = []
    if select_model == 1:
        predict_one_hot = model_together_final.predict(np.array(struct_verify_vec))
    elif select_model == 2:
        predict_one_hot = model_together_final.predict(np.array(semantic_verify_vec))
    elif select_model == 3:
        predict_one_hot = model_together_final.predict([np.array(syntatic_verify_vec),
                                                        np.array(semantic_verify_vec),
                                                        np.array(call_graph_syntatic_verify_vec),
                                                        np.array(call_graph_semantic_verify_vec),np.array(call_graph_num_verify_vec)
                                                        ])
    elif select_model == 4:
        predict_one_hot = model_together_final.predict(np.array(syntatic_verify_vec))
    elif select_model == 5:
        predict_one_hot = model_together_final.predict([np.array(call_graph_syntatic_verify_vec),
                                                        np.array(call_graph_semantic_verify_vec),
                                                        np.array(call_graph_num_verify_vec)
                                                        ])
    elif select_model == 6:
        predict_one_hot = model_together_final.predict([np.array(syntatic_verify_vec),
                                                        np.array(semantic_verify_vec)
                                                        ])
    predict_number = [np.argmax(one_hot) for one_hot in predict_one_hot]
    verify_number = [np.argmax(one_hot) for one_hot in struct_verify_labels]

    import sklearn as sk
    from sklearn.metrics import balanced_accuracy_score

    y_true = verify_number
    y_pred = predict_number

    # True Positive （真正, TP）被模型预测为正的正样本；可以称作判断为真的正确率
    print('')
    print('TP：')
    for y_true_one, y_pre_one, verify_leaf_one in zip(y_true, y_pred, struct_verify_leaf):
        if y_true_one == 1 and y_pre_one == 1:
            print(str(verify_leaf_one), end=" ", )
    # #True Negative（真负 , TN）被模型预测为负的负样本 ；可以称作判断为假的正确率
    print('')
    print('TN：')
    for y_true_one, y_pre_one, verify_leaf_one in zip(y_true, y_pred, struct_verify_leaf):
        if y_true_one == 0 and y_pre_one == 0:
            print(str(verify_leaf_one), end=" ", )
    # #False Positive （假正, FP）被模型预测为正的负样本；可以称作误报率
    print('')
    print('FP：')
    for y_true_one, y_pre_one, verify_leaf_one in zip(y_true, y_pred, struct_verify_leaf):
        if y_true_one == 0 and y_pre_one == 1:
            print(str(verify_leaf_one), end=" ", )
    # #False Negative（假负 , FN）被模型预测为负的正样本；可以称作漏报率
    print('')
    print('FN：')
    for y_true_one, y_pre_one, verify_leaf_one in zip(y_true, y_pred, struct_verify_leaf):
        if y_true_one == 1 and y_pre_one == 0:
            print(str(verify_leaf_one), end=" ", )
    print('')
    balance_accuracy, precision, recall, f1_value = plot_value(y_true, y_pred)
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    draw_confusion_matrix(verify_number, predict_number, [0, 1])
    predict_true = 0
    tree_all = 0
    for y_true_one, y_pre_one, verify_leaf_one in zip(y_true, y_pred, struct_verify_leaf):
        if y_true_one == 1 :
            tree_all += 1
            if y_pre_one == 1:
                predict_true += 1
    print('性能指标',tree_all,predict_true, balance_accuracy, precision, recall, f1_value)
    print_message("结束")
