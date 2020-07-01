import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]:  # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    
    cv2.imwrite("./output.jpg", frame)
    return frame


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--face_detection", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Input type that is goint to be used \'video\',"
                        " \'image\' or \'cam\'.")
    parser.add_argument("-if", "--input_file", required=False, type=str,
                        help="Input file for using as inference file")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections faces"
                        "(0.5 by default)")
    return parser


def infer_on_stream(args):
    input_feeder = InputFeeder(args.input_type, args.input_file)
    input_feeder.load_data()
    face_detection = Model_Face_Detection(args.face_detection)
    face_detection.load_model()
    frame = next(input_feeder.next_batch())
    face_detection.predict(frame)
    result = face_detection.check_model()

    # Draw face and write an image in the disk
    # drawed_frame = draw_boxes(
    #     frame, result, args, int(input_feeder.cap.get(3)),
    #     int(input_feeder.cap.get(4)))

    print(result)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference
    infer_on_stream(args)


if __name__ == '__main__':
    main()
