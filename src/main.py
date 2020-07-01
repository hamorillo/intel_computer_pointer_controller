import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_LandMarks_Detection


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--face_detection", required=True, type=str,
                        help="Path to the face detection model.")
    parser.add_argument("-flm", "--landmark_detection_model", required=True,
                        type=str, help="Path to the landmark detection model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Input type that is goint to be used \'video\',"
                        " \'image\' or \'cam\'.")
    parser.add_argument("-if", "--input_file", required=False, type=str,
                        help="Input file for using as inference file")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections faces"
                        "(0.5 by default)")
    parser.add_argument("-d", "--debug", type=bool, default=False,
                        help="Debug mode print and write images")
    return parser


def infer_on_stream(args):
    input_feeder = InputFeeder(args.input_type, args.input_file)
    input_feeder.load_data()

    face_detection = Model_Face_Detection(args.face_detection)
    landmark_model = Model_Facial_LandMarks_Detection(
        args.landmark_detection_model)

    face_detection.load_model()
    landmark_model.load_model()

    frame = next(input_feeder.next_batch())
    face_detection.predict(frame)
    result = face_detection.check_model()

    # Draw face and write an image in the disk
    frame, face_boxes = face_detection.preprocess_output(
        result, frame, args.prob_threshold, args.debug)

    for face_box in face_boxes:
        face = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
        landmark_model.predict(face)
        landmarks_result = landmark_model.check_model()
        eyes_frame, left_eye, right_eye = landmark_model.preprocess_output(
            landmarks_result, face, args.debug)


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
