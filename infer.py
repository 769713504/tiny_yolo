# 预测
import os
import numpy as np
import time
import paddle.fluid as fluid
from IPython.display import display
from PIL import Image
from PIL import ImageDraw

train_params = {
    "data_dir": "./img/",  # 数据目录
    "train_list": "train.txt",  # 训练集文件
    "eval_list": "eval.txt",  # 评估数据集
    "class_dim": -1,
    "label_dict": {},  # 标签字典
    "num_dict": {},
    "image_count": -1,
    "continue_train": True,  # 是否加载前一次的训练参数，接着训练
    "pretrained": False,  # 是否预训练
    "pretrained_model_dir": "./pretrained-model",
    "save_model_dir": "./yolo-model",  # 增量模型保存目录
    "model_prefix": "yolo-v3",  # 模型前缀
    "freeze_dir": "freeze_model",  # 模型固化目录(真正执行预测的模型)
    "use_tiny": True,  # 是否使用精简版YOLO模型
    "max_box_num": 20,  # 一幅图上最多有多少个目标
    "num_epochs": 40,  # 训练轮次
    "train_batch_size": 32,  # 对于完整yolov3，每一批的训练样本不能太多，内存会炸掉；如果使用tiny，可以适当大一些
    "use_gpu": False,  # 是否使用GPU
    "yolo_cfg": {  # YOLO模型参数
        "input_size": [3, 448, 448],  # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
        "anchors": [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240],  # 锚点??
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },
    "yolo_tiny_cfg": {  # YOLO tiny 模型参数
        "input_size": [3, 256, 256],
        "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],  # 数据增强使用的灰度值
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,  # 是否做图像扭曲增强
    "nms_top_k": 300,
    "nms_pos_k": 300,
    "valid_thresh": 0.01,
    "nms_thresh": 0.45,  # 非最大值抑制阈值
    "image_distort_strategy": {  # 图像扭曲策略
        "expand_prob": 0.5,  # 扩展比率
        "expand_max_ratio": 4,
        "hue_prob": 0.5,  # 色调
        "hue_delta": 18,
        "contrast_prob": 0.5,  # 对比度
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,  # 饱和度
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,  # 亮度
        "brightness_delta": 0.125},

    "sgd_strategy": {  # 梯度下降配置
        "learning_rate": 0.002,
        "lr_epochs": [30, 50, 65],  # 学习率衰减分段（3个数字分为4段）
        "lr_decay": [1, 0.5, 0.25, 0.1]  # 每段采用的学习率，对应lr_epochs参数4段
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 2.5,
        "min_curr_map": 0.84
    }
}


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数  标签字典{'apple':0,'banana':1} 与 编号字典{0:'apple',1:'banana'}
    :return:
    """
    # 训练集文件路径
    file_list = os.path.join(train_params['data_dir'], train_params['train_list'])
    # 类别文件路径
    label_list = os.path.join(train_params['data_dir'], "label_list")
    index = 0
    # codecs是专门用作编码转换通用模块
    with open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            # 将每行类别存入编号字典  {0:'apple',1:'banana'}
            train_params['num_dict'][index] = line.strip()
            # 标签字典  {'apple':0,'banana':1}
            train_params['label_dict'][line.strip()] = index
            index += 1
        # 记录一共有几类
        train_params['class_dim'] = index
    # 打开文件路径
    with open(file_list, encoding='utf-8') as flist:
        # 行列表
        lines = [line.strip() for line in flist]
        # 保存行数(图片数量)
        train_params['image_count'] = len(lines)


init_train_parameters()
ues_tiny = train_params['use_tiny']
yolo_config = train_params['yolo_tiny_cfg'] if ues_tiny else train_params['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']
label_dict = train_params['num_dict']
class_dim = train_params['class_dim']
print("标签字典:{} 类别数量:{}".format(label_dict, class_dim))

place = fluid.CUDAPlace(0) if train_params['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
# 预测模型
path = train_params['freeze_dir']
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


# 给图片画上外接矩形框
def draw_bbox_image(img, boxes, labels, save_name):
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)  # 图像绘制对象
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')  # 绘制矩形
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0))  # 绘制标签
    img.save(save_name)
    display(img)


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    # 打开图片
    origin = Image.open(img_path)
    # 调整大小
    img = resize_img(origin, target_size)
    # 拷贝
    resized_img = img.copy()
    # 不是rgb转为rgb
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 转置(交换维度数据)
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    #返回 (原图,归一化图,归一化的副本)
    return origin, img, resized_img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    # 读图:返回原图,缩放后的图片,拷贝缩放图.(输入图片路径)
    origin, tensor_img, resized_img = read_image(image_path)
    # 获取原图的宽高
    input_w, input_h = origin.size[0], origin.size[1]
    # 设置图片的形状
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))

    t1 = time.time()
    # 执行预测
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    print("预测耗时:{0}".format("%2.4f sec" % period))
    bboxes = np.array(batch_outputs[0])  # 预测结果
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("在{}中没有创建对象!".format(image_path))
        return
    labels = bboxes[:, 0].astype('int32')  # 类别
    scores = bboxes[:, 1].astype('float32')  # 概率
    boxes = bboxes[:, 2:].astype('float32')  # 边框

    print('类别:{},概率:{},边框:{}'.format(labels,scores,boxes))
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-result.jpg'
    # 在图片中绘制预测边框(原图,边框,标签,输出路径)
    draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    image_path = "./test.jpg"
    infer(image_path)
