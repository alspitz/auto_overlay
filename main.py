#!/usr/bin/env python3
import argparse

from auto_overlay import AutoOverlay

parser = argparse.ArgumentParser()
parser.add_argument('videopath', type=str, help="Video file to process")
parser.add_argument('--start-time', default=0.0, type=float, help="Video start time")

args = parser.parse_args()

args.crop_x = 0
args.crop_y = 0
args.crop_width = 3840
args.crop_height = 2160

args.rect_width = 80
args.rect_height = 80
args.frame_period = 40
args.show_rect = True

ao = AutoOverlay()

ao.auto_overlay(**vars(args))
