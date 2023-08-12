from collections import defaultdict

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import paddle.fluid.layers as layers
from tqdm import tqdm


def load_all():
    num_user, num_item = 0, 0
    file_path = 'D:/PycharmProjects/NCF_Paddle/ml-100k/u.data'
    train_mat, test_mat = {}, {}
    train_data, test_data = defaultdict(list), defaultdict(list)
    with open(file_path) as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        slices = line.split('\t')
        u = int(slices[0])
        v = int(slices[1])
        num_user = max(num_user, u)
        num_item = max(num_item, v)
        if i < 80000:
            train_mat[(u, v)] = 1
            train_data[u].append(v)
            test_mat[(u, v)] = 1
        else:
            test_mat[(u, v)] = 1
            test_data[u].append(v)
        i += 1
    return train_data, train_mat, test_data, test_mat, num_user + 1, num_item + 1


train_data, train_mat, test_data, test_mat, num_user, num_item = load_all()


def generate_test(test_data, test_mat, num_item, num_neg):
    test_negative_data = []
    for u in test_data:
        for v in test_data[u]:
            users_tmp = []
            items_tmp = []
            users_tmp.append(u)
            items_tmp.append(v)
            for i in range(num_neg):
                j = np.random.randint(num_item)
                while (u, j) in test_mat:
                    j = np.random.randint(num_item)
                users_tmp.append(u)
                items_tmp.append(j)

            dict_tmp = {
                'user': np.array(users_tmp),
                'item': np.array(items_tmp)
            }
            test_negative_data.append(dict_tmp)
    return test_negative_data


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def metrics(model, test_dict, top_k):
    HR, NDCG = [], []
    for c in test_dict:

        user = fluid.dygraph.to_variable(np.array(c['user']).astype('int'))
        item = fluid.dygraph.to_variable(np.array(c['item']).astype('int'))
        predictions = fluid.layers.reshape(model(user, item), shape=[1, -1])
        _, indices = fluid.layers.topk(predictions, top_k)
        indices = indices.numpy().tolist()[0]
        recommends = []
        for it in indices:
            recommends.append(c['item'][it])
        gt_item = c['item'][0]
        HR.append(hit(gt_item, recommends))
    return np.mean(HR)


class NCFData():
    def __init__(self, data, mat, num_user, num_item):
        self.data = data
        self.mat = mat
        self.num_user = num_user
        self.num_item = num_item

    def gt_neg(self):

        self.U = []
        self.V_pos = []
        self.V_neg = []
        for u in self.data:
            for v in self.data[u]:
                self.U.append(u)
                self.V_pos.append(v)
                j = np.random.randint(self.num_item)
                while (u, j) in self.mat:
                    j = np.random.randint(self.num_item)
                self.V_neg.append(j)

    def reader_createor(self):
        def reader():
            for i in range(len(self.U)):
                yield self.U[i], self.V_pos[i], self.V_neg[i]

        return reader


class NCF(fluid.dygraph.Layer):
    def __init__(self, user_num, item_num, factor_num, num_layers):
        super(NCF, self).__init__()
        # 构建嵌入层
        self.embed_user_MLP = dygraph.Embedding(size=[user_num, factor_num * (2 ** (num_layers - 1))],
                                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01))
        self.embed_item_MLP = dygraph.Embedding(size=[item_num, factor_num * (2 ** (num_layers - 1))],
                                                param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01))
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            # 设置隐藏层全连接
            MLP_modules.append(dygraph.Linear(input_size, input_size // 2, act='relu',
                                              param_attr=fluid.initializer.Xavier(uniform=True)))
        self.MLP_layers = fluid.dygraph.Sequential(*MLP_modules)
        predict_size = factor_num
        self.predict_layer = dygraph.Linear(predict_size, 1, param_attr=fluid.initializer.MSRAInitializer(uniform=True))

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = layers.concat((embed_user_MLP, embed_item_MLP), 1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction


train_data = NCFData(train_data, train_mat, num_user, num_item)
train_data.gt_neg()

train_reader = paddle.batch(
    reader=train_data.reader_createor(), batch_size=256
)


def Loss_func(predict_pos, predict_neg):
    loss = - fluid.layers.reduce_sum(fluid.layers.log(fluid.layers.sigmoid((predict_pos - predict_neg))))
    return loss


test_dict = generate_test(test_data, test_mat, num_item, 1000)


def train(epoch, lr):
    with fluid.dygraph.guard():
        ncf = NCF(num_user, num_item, 32, 2)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=ncf.parameters())
        for ep in range(epoch):
            loss_sum = 0.0
            ncf.train()
            for data in tqdm(train_reader(), ncols=100, colour='blue'):
                user = fluid.dygraph.to_variable(np.array([x[0] for x in data]).astype('int'))
                item_p = fluid.dygraph.to_variable(np.array([x[1] for x in data]).astype('int'))
                item_n = fluid.dygraph.to_variable(np.array([x[2] for x in data]).astype('int'))
                pred_p = ncf.forward(user, item_p)
                pred_n = ncf.forward(user, item_n)
                loss = Loss_func(pred_p, pred_n)
                loss_sum += loss
                loss.backward()
                optimizer.minimize(loss)
                ncf.clear_gradients()
            ncf.eval()
            Hr = metrics(ncf, test_dict, 10)
            print("epoch{} loss_train:{},hr:{}".format(ep, loss_sum.numpy()[0], Hr))


train(10, 0.00001)
