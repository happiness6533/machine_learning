import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yoloProject.yoloUtils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yoloProject.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    # box_confidence : tensor : (19, 19, 5, 1) : 박스 하나에 오브젝트가 존재할 확률
    # boxes : tensor : (19, 19, 5, 4) : 박스 하나의 좌표계
    # box_class_probs : tensor : (19, 19, 5, 80) : 박스 하나에 인지되는 80개 각 클래스의 확률
    # threshold : 박스 하나에 어떤 클래스가 존재할 확률 < threshhold : 제거

    # return
    # scores : tensor : (None,) : containing the class probability score for selected boxes :
    # boxes : tensor : (None, 4) : containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    # classes : tensor : (None,) : containing the index of the class detected by the selected boxes
    # None : the number of selected boxes : it depends on the threshold

    # 각각의 박스(19 * 19 * 5개)에 각 클래스가 존재할 확률
    box_scores = tf.multiply(box_confidence, box_class_probs)

    # 각 박스(19 * 19 * 5개)에 존재하는 각 클래스(80개)중에서 최대 확률
    # >> 따라서 이제 각 박스는 단 1개의 클래스에 대한 최대 확률값만 가지고 있다
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)

    # 최대 확률이 threshhold보다 작다면, 그 박스는 0으로 리턴
    # 최대 확률이 threshhold보다 크다면, 그 박스는 1로 리턴
    # 즉, 각 클래스에서 최대확률이 0.6이 넘는 확률을 보유한 박스만 살아남는다
    filtering_mask = (box_class_scores >= threshold)

    # argmax : 최대값을 가지는 인덱스를 반환한다
    # (19 * 19 * 5개)의 각 박스에서 80개의 클래스가 가지는 각 확률들 중 최대값을 가지는 클래스 넘버만 리턴한다
    # 즉, (19 * 19 * 5)개의 각 박스에 >> 최대확률인 클래스의 [인덱스]를 리턴
    box_classes = K.argmax(box_scores, axis=-1)

    # 각 박스에서 최대 확률의 클래스가 가지는 확률이 0.6이 넘어가는 값만 살아남는 텐서
    scores = tf.boolean_mask(box_class_scores, filtering_mask, name="scores")
    # 박스의 좌표계들 중에서, 필터링을 거치는 박스들의 좌표들만 살아남는 텐서
    boxes = tf.boolean_mask(boxes, filtering_mask, name="boxes")
    # 각 박스의 최대확률을 가지는 인덱스 중에서, 필터링을 거친 인덱스만 살아남는 텐서
    classes = tf.boolean_mask(box_classes, filtering_mask, name="classes")

    return scores, boxes, classes


# 이미 텐서플로우에 구현이 되어있어서 사용할 일이 없다
def iou(box1, box2):
    # box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    # box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


# 위에 있는 iou함수를 사용할 필요가 없이, 텐서플로우에 이미 논맥스 서프레션 알고리즘이 구현되어 있다
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    # return
    # scores -- tensor of shape (, None), predicted score for each box
    # boxes -- tensor of shape (4, None), predicted box coordinates
    # classes -- tensor of shape (, None), predicted class for each box

    # Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    # function will transpose the shapes of scores, boxes, classes. This is made for convenience.

    # 정수 10을 텐서에서 사용하기 위해서, 10을 텐서형으로 선언
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    # 사용하려면, 이 변수를 초기화해야한다
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # 논맥스서프레션1
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold, name="nms_indices")

    # 논맥스서프레션2
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5)

    return scores, boxes, classes


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """
    # 데이터 처리 : 오리지날 이미지 + 데이터로 변환된 이미지 리턴
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    plt.show()

    return out_scores, out_boxes, out_classes


sess = K.get_session()

class_names = read_classes("data/coco_classes.txt")
anchors = read_anchors("data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes = predict(sess, "0114.jpg")
