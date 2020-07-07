import cv2
import statistics
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_LandMarks_Detection
from head_pose_estimation import Model_Head_Pose_Estimation
from gaze_estimation import Model_Gaze_Estimation


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--face_detection", required=True, type=str,
                        help="Path to face detection model.")
    parser.add_argument("-flm", "--landmark_detection_model", required=True,
                        type=str, help="Path to landmark detection model.")
    parser.add_argument("-hpm", "--head_pose_model", required=True,
                        type=str, help="Path to head pose estimation model.")
    parser.add_argument("-gem", "--gaze_estimation_model", required=True,
                        type=str, help="Path to head gaze estimation model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Input type that is going to be used \'video\',"
                        " \'image\' or \'cam\'.")
    parser.add_argument("-if", "--input_file", required=False, type=str,
                        help="Input file for using as inference file")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections faces"
                        "(0.5 by default)")
    parser.add_argument("-d", "--debug", type=bool, default=False,
                        help="Debug mode print and write images")
    return parser


def print_statistics(face_detection, landmark_model, head_pose_model,
                     gaze_estimation_model):
    print("Face Detection Average Infer Time:" +
          str(statistics.mean(face_detection.infer_times))+"ms")
    print("Landmark Average Infer Time:" +
          str(statistics.mean(landmark_model.infer_times))+"ms")
    print("Head Pose Average Infer Time:" +
          str(statistics.mean(head_pose_model.infer_times))+"ms")
    print("Gaze Average Infer Time:" +
          str(statistics.mean(gaze_estimation_model.infer_times))+"ms")


def infer_on_stream(args):

    mc = MouseController('medium', 'fast')
    input_feeder = InputFeeder(args.input_type, args.input_file)
    input_feeder.load_data()

    face_detection = Model_Face_Detection(args.face_detection)
    landmark_model = Model_Facial_LandMarks_Detection(
        args.landmark_detection_model)
    head_pose_model = Model_Head_Pose_Estimation(args.head_pose_model)
    gaze_estimation_model = Model_Gaze_Estimation(args.gaze_estimation_model)

    face_detection.load_model()
    landmark_model.load_model()
    head_pose_model.load_model()
    gaze_estimation_model.load_model()

    while(True):
        frame = next(input_feeder.next_batch())

        if(frame is None):
            break

        face_detection.predict(frame)
        result = face_detection.check_model()

        # Draw face and write an image in the disk
        frame, face_boxes = face_detection.preprocess_output(
            result, frame, args.prob_threshold, args.debug)

        for face_box in face_boxes:
            face = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            landmark_model.predict(face)
            head_pose_model.predict(face)

            landmarks_result = landmark_model.check_model()
            eyes_frame, left_eye, right_eye = landmark_model.preprocess_output(
                landmarks_result, face, args.debug)

            head_pose_result = head_pose_model.check_model()
            gaze_estimation_model.predict(
                face, left_eye, right_eye, head_pose_result, args.debug)
            gaze_result = gaze_estimation_model.check_model()

            mc.move(gaze_result['gaze_vector'][0][0],
                    gaze_result['gaze_vector'][0][1])

    print_statistics(face_detection, landmark_model, head_pose_model,
                     gaze_estimation_model)


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
