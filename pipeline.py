import json
import os
import time
import argparse
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
from craft_detection import craft_utils
from craft_detection import imgproc
from craft_detection import file_utils
from craft_detection.craft import CRAFT
from scene_text_recognition.model import Model as STRModel
from scene_text_recognition.utils import CTCLabelConverter, AttnLabelConverter
import string
from PIL import Image
from scene_text_recognition.dataset import ResizeNormalize
from craft_detection.test import copyStateDict, str2bool
from utils import extract_file_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio):
    t0 = time.time()

    # resize
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR,
                                                               mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1
    print(t1)

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_craft_model', default='./CRAFT_models/craft_mlt_25k.pth', type=str,
                        help='trained_craft_model')
    parser.add_argument('--trained_str_model', default='./STR_models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth',
                        type=str, help='trained_str_model')

    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--data', default='./data/', type=str, help='folder path to input images')
    parser.add_argument('--result_dir', default='./result/', type=str, help='folder path to load result images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', default=False, help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', default=True, help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(opt.data)

    result_folder = opt.result_dir
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    craft_model = CRAFT()

    print('Loading CRAFT weights from checkpoint (' + opt.trained_craft_model + ')')
    print('Loading STR weights from checkpoint (' + opt.trained_str_model + ')')

    str_model = STRModel(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    str_model = torch.nn.DataParallel(str_model).to(device)
    if opt.cuda:
        craft_model.load_state_dict(copyStateDict(torch.load(opt.trained_craft_model)))
        str_model.load_state_dict(torch.load(opt.trained_str_model))
    else:
        craft_model.load_state_dict(copyStateDict(torch.load(opt.trained_craft_model, map_location='cpu')))
        str_model.load_state_dict(torch.load(opt.trained_str_model, map_location="cpu"))

    if opt.cuda:
        craft_model = craft_model.cuda()
        craft_model = torch.nn.DataParallel(craft_model)
        cudnn.benchmark = False

    craft_model.eval()
    str_model.eval()

    # load data

    for k, image_path in enumerate(image_list):
        start_time = time.time()
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        img = imgproc.loadImage(image_path)

        bboxes, polys, score_text = predict(craft_model, img, opt.text_threshold, opt.link_threshold, opt.low_text, opt.cuda,
                                            opt.poly, opt.canvas_size, opt.mag_ratio)
        images = []
        regions = []
        start = time.time()
        for bbox in bboxes:
            x, y = bbox[0]
            x, y = int(x), int(y)
            w, h = bbox[2] - bbox[0]
            w, h = int(w), int(h)
            sub_image = img[y:y + h, x:x + w]
            sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
            try:
                sub_image = Image.fromarray(sub_image)
            except AttributeError:
                print(f"Error with region {(x, y, w, h)}, skipping")
                continue
            images.append(sub_image)
            regions.append((x, y, w, h))

        transform = ResizeNormalize((opt.imgW, opt.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        batch_size = len(images)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = str_model(image_tensors, text_for_pred)
            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = str_model(image_tensors, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
        results = defaultdict(lambda: [])
        for image, pred, region in zip(images, preds_str, regions):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
            results[pred].append(region)
        file_name = extract_file_name(opt.data, image_path)
        with open(f'{opt.result_dir}/{file_name}.json', 'w') as fp:
            json.dump(results, fp)

        end_time = time.time()
        print("Duration for one image: " + str(end_time - start_time) + "s", flush=True)
