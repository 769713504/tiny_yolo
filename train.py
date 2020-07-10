# 日志相关配置
import json
import logging
import os
import random
import time

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from PIL import Image, ImageEnhance

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
    "num_epochs": 10,  # 训练轮次
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


class YOLOv3Tiny(object):
    def __init__(self, class_num, anchors, anchor_mask):
        """
        YOLOv3Tiny构造方法
        :param class_num: 类别数量
        :param anchors: 锚点
        :param anchor_mask: 锚框
        """
        self.outputs = []  # 该模型的输出
        self.downsample_ratio = 1  # 下采样率
        self.anchor_mask = anchor_mask  # 锚框
        self.anchors = anchors  # 锚点
        self.class_num = class_num  # 类别数量

        self.yolo_anchors = []
        self.yolo_classes = []

        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])

            print("mask_anchors:", mask_anchors)

            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3-tiny'

    # 获取锚点
    def get_anchors(self):
        return self.anchors

    # 获取锚框
    def get_anchor_mask(self):
        return self.anchor_mask

    # 获取类别编号
    def get_class_num(self):
        return self.class_num

    # 获取降采样率
    def get_downsample_ratio(self):
        return self.downsample_ratio

    # 获取yolo锚点列表
    def get_yolo_anchors(self):
        return self.yolo_anchors

    # 获取yolo类别列表
    def get_yolo_classes(self):
        return self.yolo_classes

    # 带batch_normal的卷积层,
    # batch_normal: 归一化,减去均值,单位化方差,这样训练更快,因为把数据集都映射到原点周围了。但是这样会导致后面激活函数的表达能力变差,所以作者又引入了缩放和平移,
    def conv_bn(self, input, num_filters, filter_size, stride, padding, num_groups=1, use_cudnn=True):

        '''conv_bn(self,输入,  卷积核数量,      卷积核大小,   步长,    填充,    组数量,        是否是用use_cudnn对cuda加速'''

        # 参数属性
        param_attr = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.2))
        # 卷积层
        conv = fluid.layers.conv2d(
            input=input,  # 输入
            num_filters=num_filters,  # 卷积核数量
            filter_size=filter_size,  # 卷积核大小
            stride=stride,  # 步长
            padding=padding,  # 填充
            act=None,  # 激活函数
            groups=num_groups,
            use_cudnn=use_cudnn,  # 是否是用use_cudnn对cuda加速
            param_attr=param_attr,  # 参数初始值
            bias_attr=False)  # 输出中不带偏置
        # Batch Noraml操作
        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        # 在batch_norm中使用leaky的话，只能使用默认的alpha=0.02；如果需要设值，必须提出去单独来
        # 正则化的目的，是为了防止过拟合，较小的L2值能防止过拟合

        # 参数属性:加入正则项,防止过拟合
        param_attr = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.2), regularizer=L2Decay(0.0))
        # 偏差属性
        bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.0))
        # 获取输出:传入(卷积网络,参数属性,偏差属性)
        out = fluid.layers.batch_norm(input=conv, param_attr=param_attr, bias_attr=bias_attr)
        return out
        # 深度可分离卷积;(self, 输入, 滤波器大小, 步长, 填充)

    # 深度可分离卷积层
    def depthwise_conv_bn(self, input, filter_size=3, stride=1, padding=1):
        num_filters = input.shape[1]  # 取出通道数量
        # print("深度可分离卷积的滤波器数量:depthwise_conv_bn.num_filters:", num_filters)
        return self.conv_bn(input, num_filters=num_filters, filter_size=filter_size, stride=stride,
                            padding=padding, num_groups=num_filters)

    # 降采样(池化)
    def down_sample(self, input, pool_size=2, pool_stride=2):
        self.downsample_ratio *= 2  # 记录降采样率
        return fluid.layers.pool2d(input=input, pool_type="max", pool_size=pool_size, pool_stride=pool_stride)

    # 基本块,包含一个卷积、池化操作:(输入,滤波器数量)
    def basic_block(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)  # 卷积
        out = self.down_sample(conv1)  # 池化
        return out

    # 上采样:图像或矩阵放大,(输入,放大倍率)
    def up_sample(self, input, scale=2):  # 上采样率
        shape_nchw = fluid.layers.shape(input)  # 获取输入形状:大小,数目,高度,宽度
        # print("上采样的 大小,数目,高度,宽度为.up_sample.shape_nchw:", shape_nchw)
        # 取出输入数据的高度和宽度,传入(形状, 行, 开始索引, 结束索引)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype="int32")
        out_shape = in_shape * scale  # 计算输出形状
        out_shape.stop_gradient = True
        # 上采样操作(最邻近插值法)
        out = fluid.layers.resize_nearest(input=input, scale=scale, actual_shape=out_shape)
        return out

    # yolo检测模块(输入图,滤波器数量)
    def yolo_detection_block(self, input, num_filters):
        # 网络 (输入,滤波器数量,滤波器大小,步长,填充)
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        # 执行第二层网络
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip  # 返回第一层 第二层对象

    # 组建YOLO网络模型（精简版YOLO3模型）
    def net(self, img):
        # YOLO精简版没有残差层，只使用了两个不同尺度的输出层
        stages = [16, 32, 64, 128, 256, 512]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than down_sample times"
        tmp = img
        blocks = []  # 装载每一层的列表
        # 循环创建卷积池化组
        for i, stage_count in enumerate(stages):
            if i == len(stages) - 1:  # 最后一组(后跟3个卷积)
                # 首先是一组卷积池化
                block = self.conv_bn(tmp, stage_count, filter_size=3, stride=1, padding=1)
                blocks.append(block)
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.conv_bn(blocks[-1], stage_count * 2, filter_size=1, stride=1, padding=0)
                blocks.append(block)
            else:  # 不是最后一组
                tmp = self.basic_block(tmp, stage_count)
                blocks.append(tmp)

        blocks = [blocks[-1], blocks[3]]  # 取出其中两层

        # 跨视域处理
        for i, block in enumerate(blocks):
            if i > 0:  # 不是第一层，和前面的层做跨视域链接
                block = fluid.layers.concat(input=[route, block], axis=1)
            if i < 1:  # i=0时，创建两个卷基层
                route, tip = self.yolo_detection_block(block, num_filters=(256 // (2 ** i)))
            else:
                tip = self.conv_bn(block, num_filters=256,
                                   filter_size=3,
                                   stride=1,
                                   padding=1)
            # 初始化一组权重偏置
            param_attr = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.2))
            bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0))

            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=param_attr,
                bias_attr=bias_attr)
            self.outputs.append(block_out)
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.up_sample(route)

        return self.outputs


def init_log_config():  # 初始化日志相关配置
    global logger
    # 创建日志对象
    logger = logging.getLogger()
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 拼接日志的保存路径
    log_path = os.path.join(os.getcwd(), 'logs')
    # 若路径不存在则创建路径
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 拼接训练日志文件路径
    log_name = os.path.join(log_path, 'train.log')
    # 打开文件句柄
    fh = logging.FileHandler(log_name, mode='w')
    # 设置级别
    fh.setLevel(logging.DEBUG)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # 将格式保存到句柄
    fh.setFormatter(formatter)
    # 给日志对象添加句柄
    logger.addHandler(fh)


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


def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    if is_tiny:
        return YOLOv3Tiny(class_num, anchors, anchor_mask)
    else:
        pass


def build_program_with_feeder(main_prog, startup_prog, place):
    # 最大选框数量
    max_box_num = train_params['max_box_num']
    # 获取是否使用tiny yolo参数
    ues_tiny = train_params['use_tiny']
    # 配置yolo参数
    yolo_config = train_params['yolo_tiny_cfg'] if ues_tiny else train_params['yolo_cfg']
    # 更改全局主程序和启动程序
    with fluid.program_guard(main_prog, startup_prog):
        # 设置图片: (图片名称,  形状, 类型)
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')  # 图像
        # 获取边框: (名称,形状,类型)
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')  # 边框
        # 获取标签
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')  # 标签
        # 数据喂入器(喂入列表,执行位置,主程序)
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=main_prog)  # 定义feeder
        # 数据读取器(训练列表文件路径,数据文件夹路径,输入尺寸,模式)
        reader = single_custom_reader(train_params['train_list'], train_params['data_dir'], yolo_config['input_size'],
                                      'train')  # 读取器
        # 获取ues_tiny状态: 是否开启了ues_tiny
        ues_tiny = train_params['use_tiny']
        # 获取配置信息
        yolo_config = train_params['yolo_tiny_cfg'] if ues_tiny else train_params['yolo_cfg']

        with fluid.unique_name.guard():
            # 创建yolo模型:(训练参数,锚点,锚框)-->yolo类的对象模型
            model = get_yolo(ues_tiny, train_params['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
            # 输出网络
            outputs = model.net(img)
        # 返回(喂入器,阅读器,损失值)
        return feeder, reader, get_loss(model, outputs, gt_box, gt_label)


def single_custom_reader(file_path, data_dir, input_size, mode):
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    reader = custom_reader(images, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_params['train_batch_size'])
    reader = paddle.batch(reader, train_params['train_batch_size'])
    # 生成器?
    return reader


def custom_reader(file_list, data_dir, input_size, mode):
    def reader():
        np.random.shuffle(file_list)  # 打乱文件列表

        for line in file_list:  # 读取行，每行一个图片及标注
            if mode == 'train' or mode == 'eval':
                ######################  以下可能是需要自定义修改的部分   ############################
                parts = line.split('\t')  # 按照tab键拆分
                image_path = os.path.join(data_dir, parts[0])
                # 读取图像数据
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # bbox 的列表，每一个元素为这样
                # layout: label | x-center | y-cneter | width | height | difficult
                bbox_labels = []
                for object_str in parts[1:]:  # 循环处理每一个目标标注信息
                    if len(object_str) <= 1:
                        continue
                    bbox_sample = []
                    object = json.loads(object_str)
                    bbox_sample.append(float(train_params['label_dict'][object['value']]))
                    bbox = object['coordinate']  # 获取框坐标
                    # 计算x,y,w,h
                    box = [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                    bbox = box_to_center_relative(box, im_height, im_width)  # 坐标转换
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    difficult = float(0)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                ######################  可能需要自定义修改部分结束   ############################
                if len(bbox_labels) == 0:
                    continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)  # 预处理
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0:
                    continue
                boxes = sample_labels[:, 1:5]  # 坐标
                lbls = sample_labels[:, 0].astype('int32')  # 标签
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_params['max_box_num']  # 一副图像最多多少个目标物体
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)  # 控制最大目标数量
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]
                yield img, ret_boxes, ret_lbls  # 返回图像、边框、标签
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader


def box_to_center_relative(box, img_height, img_width):
    """
    Convert COCO annotations box with format [x1, y1, w, h] to
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box应该是一个len(4)列表或元组"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w - 1, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, img_height - 1)

    x = (x1 + x2) / 2 / img_width  # x中心坐标
    y = (y1 + y2) / 2 / img_height  # y中心坐标
    w = (x2 - x1) / img_width  # 框宽度/图片总宽度
    h = (y2 - y1) / img_height  # 框高度/图片总高度

    return np.array([x, y, w, h])


def get_loss(model, outputs, gt_box, gt_label):
    losses = []
    downsample_ratio = model.get_downsample_ratio()
    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            loss = fluid.layers.yolov3_loss(x=out,
                                            gt_box=gt_box,  # 真实边框
                                            gt_label=gt_label,  # 标签
                                            anchors=model.get_anchors(),  # 锚点
                                            anchor_mask=model.get_anchor_mask()[i],
                                            class_num=model.get_class_num(),
                                            ignore_thresh=train_params['ignore_thresh'],
                                            # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                                            use_label_smooth=False,
                                            downsample_ratio=downsample_ratio)
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer = optimizer_sgd_setting()
        optimizer.minimize(loss)
        return loss


def optimizer_sgd_setting():
    batch_size = train_params["train_batch_size"]  # batch大小
    iters = train_params["image_count"] // batch_size  # 计算轮次
    iters = 1 if iters < 1 else iters
    learning_strategy = train_params['sgd_strategy']
    lr = learning_strategy['learning_rate']  # 学习率
    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries, values),  # 分段衰减学习率
                                             regularization=fluid.regularizer.L2Decay(0.00005))  # L2权重衰减正则化

    return optimizer


def preprocess(img, bbox_labels, input_size, mode):
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)

    if mode == 'train':
        if train_params['apply_distort']:  # 是否扭曲增强
            img = distort_image(img)

        img, gtboxes = random_expand(img, sample_labels[:, 1:5])  # 扩展增强
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])  # 随机裁剪
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes

    img = resize_img(img, sample_labels, input_size)
    img = np.array(img).astype('float32')
    img -= train_params['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


def distort_image(img):  # 图像扭曲
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def random_expand(img, gtboxes, keep_ratio=True):
    if np.random.uniform(0, 1) < train_params['image_distort_strategy']['expand_prob']:
        return img, gtboxes
    max_ratio = train_params['image_distort_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)
    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_params['mean_rgb'][i]
    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [(0.1, 1.0),
                       (0.3, 1.0),
                       (0.5, 1.0),
                       (0.7, 1.0),
                       (0.9, 1.0),
                       (0.0, 1.0)]  # 最小/最大交并比值

    w, h = img.size
    crops = [(0, 0, w, h)]

    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h / float(h)
            ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels


def resize_img(img, sampled_labels, input_size):
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)  # 重置大小，双线性插值
    return img


def random_brightness(img):  # 亮度
    prob = np.random.uniform(0, 1)

    if prob < train_params['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_params['image_distort_strategy']['brightness_delta']  # 默认值0.125
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1  # 产生均匀分布随机值
        img = ImageEnhance.Brightness(img).enhance(delta)  # 调整图像亮度

    return img


def random_contrast(img):  # 对比度
    prob = np.random.uniform(0, 1)
    if prob < train_params['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_params['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):  # 饱和度
    prob = np.random.uniform(0, 1)
    if prob < train_params['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_params['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)

    return img


def random_hue(img):  # 色调
    prob = np.random.uniform(0, 1)
    if prob < train_params['image_distort_strategy']['hue_prob']:
        hue_delta = train_params['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    # 取两个框的坐标
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1  # 相交部分宽度
    inter_h = inter_y2 - inter_y1 + 1  # 相交部分高度
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h  # 相交面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  # 框1的面积
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  # 框2的面积

    return inter_area / (b1_area + b2_area - inter_area)  # 交集面积/并集面积


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


# 执行训练
def train():
    # 初始化日志
    init_log_config()
    # 初始化参数
    init_train_parameters()
    # 打印参数
    logger.info("YOLOv3, train开始训练~ 参数: %s", str(train_params))
    # 是否开启tiny
    ues_tiny = train_params['use_tiny']
    # 读取配置
    yolo_config = train_params['yolo_tiny_cfg'] if ues_tiny else train_params['yolo_cfg']
    # 获得模型  (类别数目,锚点,锚框)
    model = get_yolo(ues_tiny, train_params['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
    # 图片数据对象 (名称, 输入数据形状, 类型)
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    # 图片形状 (名称 形状 类型)
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')
    # 选择执行位置
    place = fluid.CUDAPlace(0) if train_params["use_gpu"] else fluid.CPUPlace()
    # 写日志
    logger.info("建立网络 和 程序")
    # 初始化训练程序
    train_program = fluid.Program()
    # 开始程序
    start_program = fluid.Program()
    # 喂入器, 阅读器, 损失值 = 建立程序和喂入器(主程序,开始程序,执行位置)
    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)
    # 在cpu/gpu 上创建执行对象
    exe = fluid.Executor(place)
    # 执行开始程序
    exe.run(start_program)  # 使用start_program作为启动program, 使用内在的op进行初始化
    # 训练_取得_列表 # 获取的返回值
    train_fetch_list = [loss.name]
    # 加载增量模型: 如果增量训练打开 且 模型目录存在 就进行增量训练
    if train_params['continue_train'] and os.path.exists(train_params['save_model_dir']):
        # 写日志
        logger.info('从训练模型中加载参数~')
        # 加载增量模型(程序对象, 模型的保存目录, 主程序)
        fluid.io.load_persistables(executor=exe, dirname=train_params['save_model_dir'], main_program=train_program)
    # 开始迭代训练 (设置训练轮次)
    for pass_id in range(train_params["num_epochs"]):
        # 写日志
        logger.info("第{}轮,开始读取图片.".format(pass_id))
        # 批次编号
        batch_id = 0
        # 损失值合计
        total_loss = 0.0
        # 遍历文件批量随机行读取器
        for batch_id, data in enumerate(reader()):
            # 记录时间
            t1 = time.time()
            # 损失值=运行程序(主程序,数据喂入器喂入程序,取出列表)
            loss = exe.run(train_program, feed=feeder.feed(data), fetch_list=train_fetch_list)  # 训练
            # 计算本次训练所花时间
            period = time.time() - t1
            # 计算平均损失值
            loss = np.mean(np.array(loss))
            # 累加损失值
            total_loss += loss
            # batch_id += 1
            # 20条记录一条日志
            if batch_id % 20 == 0:
                logger.info("第{}轮, 第{}批, 损失值{}, 耗时{}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
        # 计算批次平均损失值并打印
        pass_mean_loss = total_loss / batch_id
        logger.info("经过{0}轮训练, 当前训练的平均损失值为{1}".format(pass_id, pass_mean_loss))
    # 训练完成后，保存增量模型
    logger.info("训练完成~,结束训练.")
    # 获取模型保存地址
    save_model_dir = train_params['save_model_dir']
    # 如果固化模型目录不存在,创建目录
    if not os.path.exists(train_params['freeze_dir']):
        os.makedirs(train_params['freeze_dir'])
    # 持久化增量模型(保存路径, 主程序, 程序执行对象)
    fluid.io.save_persistables(executor=exe, dirname=save_model_dir, main_program=train_program)
    print("持久化增量模型完成~  保存位置:", save_model_dir)



if __name__ == '__main__':
    train()

def freeze_model():
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    ues_tiny = train_params['use_tiny']
    yolo_config = train_params['yolo_tiny_cfg'] if ues_tiny else train_params['yolo_cfg']
    path = train_params['save_model_dir']

    model = get_yolo(ues_tiny, train_params['class_dim'],
                     yolo_config['anchors'], yolo_config['anchor_mask'])
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')

    boxes = []
    scores = []
    outputs = model.net(image)
    downsample_ratio = model.get_downsample_ratio()

    for i, out in enumerate(outputs):
        box, score = fluid.layers.yolo_box(x=out,
                                           img_size=image_shape,
                                           anchors=model.get_yolo_anchors()[i],
                                           class_num=model.get_class_num(),
                                           conf_thresh=train_params['valid_thresh'],
                                           downsample_ratio=downsample_ratio,
                                           name="yolo_box_" + str(i))
        boxes.append(box)
        scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
        downsample_ratio //= 2

    pred = fluid.layers.multiclass_nms(bboxes=fluid.layers.concat(boxes, axis=1),
                                       scores=fluid.layers.concat(scores, axis=2),
                                       score_threshold=train_params['valid_thresh'],
                                       nms_top_k=train_params['nms_top_k'],
                                       keep_top_k=train_params['nms_pos_k'],
                                       nms_threshold=train_params['nms_thresh'],
                                       background_label=-1,
                                       name="multiclass_nms")

    freeze_program = fluid.default_main_program()

    fluid.io.load_persistables(exe, path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)
    # print("freeze out: {0}, pred layout: {1}".format(train_params['freeze_dir'], pred))
    # 保存模型
    if not os.path.exists(train_params['freeze_dir']):
        os.makedirs(train_params['freeze_dir'])
    fluid.io.save_inference_model(train_params['freeze_dir'],
                                  ['image', 'image_shape'],
                                  pred, exe, freeze_program)

    print("持久化增量模型完成~  保存位置:",train_params['freeze_dir'])


if __name__ == '__main__':
    freeze_model()