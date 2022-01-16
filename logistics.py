import numpy as np
import matplotlib.pyplot as plt
import json

TRAIN_DIR = "./dataset/trainset.json"
VALID_DIR = "./dataset/devset.json"
TEST_DIR = "./dataset/testset.json"
OUTPUT_DIR = "./output/testset_补充.json"

LEARNING_RATE = 0.01
BATCH_SIZE = 1024
THRESHOLD = 5000

# 加载数据集
def load_data(dirname = TRAIN_DIR):
    data = []
    with open(dirname, "r") as data_file:
        data = json.load(data_file)
    data_x = [i[0] for i in data]
    data_y = [i[1] for i in data]

    return np.array(data_x), np.array(data_y)

# 激活函数
def sigmoid(z):
    return 1 / (1+ np.exp(-z))

# 回归模型
def model(data_x, weights):
    return sigmoid(np.dot(data_x, weights.T))

# 损失值计算
def loss(data_x, data_y, weights):
    model_value = model(data_x, weights).ravel()
    left = np.multiply(-data_y, np.log(model_value))
    right = np.multiply(1 - data_y, np.log(1 - model_value))
    return np.mean(left - right)

# 梯度计算
def gradient(data_x, data_y, weights):
    grad = np.zeros(weights.shape)
    error = (model(data_x, weights)).ravel() - data_y
    for j in range(len(weights.ravel())):
        term = np.multiply(error, data_x[:, j])
        grad[0, j] = np.sum(term) / len(data_x)
    return grad

# loss曲线绘画
def draw_loss(train_loss, valid_loss):
    plt.plot(train_loss, c='b', label='train loss')
    plt.plot(valid_loss, c='r', label='valid loss')
    plt.legend(loc=0, ncol=1)
    plt.xlabel('iternations')
    plt.ylabel('loss')
    plt.savefig('./train和valid的loss曲线图.png')
    plt.show()

# 模型预测值计算
def predict(data_x, weights):
    return [1 if x > 0.5 else 0 for x in model(data_x, weights)]

# 准确率计算
def accuracy(real_val, predict_val):
    correct = [1 if real_val[i] == predict_val[i] else 0 for i in range(len(real_val))]
    return np.sum(correct) / len(real_val)

# 训练模型
def train(batchSize, threshold, learning_rate):
    batch = 0
    epoch = 0

    # 加载训练&验证集
    train_x, train_y = load_data(TRAIN_DIR)
    valid_x, valid_y = load_data(VALID_DIR)

    # 初始化weights
    weights = np.zeros((1, train_x.shape[1]))

    # 初始化loss
    train_loss = [loss(train_x, train_y, weights)]
    valid_loss = [loss(valid_x, valid_y, weights)]

    # 初始化梯度
    grad = np.zeros(weights.shape)

    # 梯度下降
    while True:
        grad = gradient(train_x[batch : batch + batchSize], train_y[batch : batch + batchSize], weights)
        # 参数更新
        weights = weights - learning_rate * grad
        train_loss.append(loss(train_x, train_y, weights))
        valid_loss.append(loss(valid_x, valid_y, weights))
        epoch += 1

        if epoch > threshold:
            break;
    
    # 画训练集&验证集的loss曲线
    draw_loss(train_loss, valid_loss)

    # 计算训练集&验证集的准确率
    print("训练集准确率：", accuracy(train_y, predict(train_x, weights)))
    print("验证集准确率：", accuracy(valid_y, predict(valid_x, weights)))
    
    return weights

# 测试模型
def test(weights):
    # 加载测试集
    test_data = []
    with open(TEST_DIR, "r") as data_file:
        test_data = json.load(data_file)
    test_data = np.array(test_data)

    # 预测测试集
    test_predict = predict(test_data, weights)

    # 输出测试结果
    new_test_data=[[test_data[i].tolist(), test_predict[i]] for i in range(len(test_predict))]
    with open(OUTPUT_DIR, "w") as file:
        json.dump(new_test_data, file)

if __name__ == '__main__':
    # 训练
    weights = train(batchSize=BATCH_SIZE, threshold=THRESHOLD, learning_rate=LEARNING_RATE)
    # 测试
    test(weights)
