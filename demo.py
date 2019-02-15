from lib.FaceDAR import FaceDAR
import cv2, os, time

#dar = FaceDAR('./config_dsfd_onet_arcface.yaml')
dar = FaceDAR('./config_fpn_onet_arcface.yaml')

dar.process_folder('./frames')