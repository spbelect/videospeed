#!/usr/bin/env python
# Usage: python3 speedup_video.py -i input.mp4 -o tmp_dir

import csv
import json
import sys
import os
import argparse
import subprocess

from collections import defaultdict, Mapping, namedtuple
from datetime import datetime
from os import makedirs
from os.path import dirname, basename, join, exists
from pathlib import Path
from subprocess import Popen, PIPE, DEVNULL, check_output as sh

import numpy as np
import cv2

from click import Context, confirm, command, option, progressbar as progress

from .merge import merge


QUIET = False

echo = lambda *a, **kw: not QUIET and print(*a, **kw)

printchar = lambda *a, **kw: echo(*a, end='', flush=True, **kw)

Context.get_usage = Context.get_help  # show full help on error


RED = (255, 0, 0)


class FFmpegVideo(object):
    def __init__(self, file_path, mode = 'r', depth = 3, pipe_buffer_size = 10 ** 8, fps = None, height=480, width=640, codec='h264', extra_args = []):
        self.file_path = file_path
        self.pipe_buffer_size = pipe_buffer_size
        self.depth = depth
        self.height = height
        self.width = width
        self.codec = codec
        self.num_frames = 0
        self.extra_args = extra_args
        self.pipe = None
        self.fps = fps

        if mode == 'r':
            self.height, self.width, self.fps_, self.num_frames = list(map(json.loads(sh(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', self.file_path]).decode('utf-8'))['streams'][0].get, ['height', 'width', 'avg_frame_rate', 'nb_frames']))
            self.num_frames = int(self.num_frames)
            if fps is None:
                self.fps = float(self.fps_.split('/')[0]) / float(self.fps_.split('/')[1])
            else:
                self.fps_ = fps

    def __iter__(self):
        self.pipe = Popen(['ffmpeg', '-i', self.file_path, '-an', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-c:v', 'rawvideo', '-r', str(self.fps_), '-'], bufsize = self.pipe_buffer_size, stdout = PIPE, stderr = open(os.devnull, 'w'))
        while self.pipe.poll() is None:
            frame = np.empty((self.height, self.width, self.depth), dtype = np.uint8)
            self.pipe.stdout.readinto(memoryview(frame))
            self.pipe.stdout.flush()
            yield frame

    def write(self, frame):
        self.pipe = self.pipe or Popen(['ffmpeg', '-nostats', '-hide_banner', '-framerate', str(self.fps), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-s:v', '{}x{}'.format(frame.shape[1], frame.shape[0]), '-i', '-', '-c:v', self.codec, '-an', '-pix_fmt', 'yuv420p', '-s:v', '{}x{}'.format(self.width, self.height)] + self.extra_args + ['-y', self.file_path], stdin = PIPE, stderr=DEVNULL)
        self.pipe.stdin.write(frame.tobytes())
        self.pipe.stdin.flush()

    def __del__(self):
        if self.pipe is not None and self.pipe.stdin is not None:
            self.pipe.stdin.close()
        del self.pipe
        self.pipe = None
        
        
class RoiMask(object):
    def __init__(self, height, width, boxes):
        polygon = [(np.clip(np.array(box, dtype=np.float32).reshape((-1, 2)), 0.0, 1.0) * np.array([width, height], dtype=np.float32)).astype(np.int32) for box in boxes]
        self.bbox = list(map(cv2.boundingRect, polygon))
        self.polygon_area = list(map(cv2.contourArea, polygon))
        full_mask = np.zeros((height, width), dtype = np.uint8)
        cv2.fillPoly(full_mask, polygon, 1)
        
        #timestamp_box_height = 40
        #cv2.rectangle(full_mask, (0, 0), (width, timestamp_box_height), 1, cv2.FILLED)
        
        self.roi_mask = [full_mask[y : y + h, x : x + w] != 1 for x, y, w, h in self.bbox]


def buffer_half(iterable, fps):
    fps = int(fps // 4)
    buf = []
    for n, elem in enumerate(iterable):
        buf.append(elem)
        if n < fps:
            continue
        if len(buf) > fps * 2 + 1:
            del buf[0]
        yield buf[-fps - 1], buf

    for i in range(fps):
        del buf[0]
        yield buf[fps], buf
            
            
def speedup(src, dst, boxes):
    """
    Speedup source file, given the boxes coordinates to detect motion.
    """
    level = 0.005
    slow, fast = 2, 16
    src = FFmpegVideo(src, 'r')
    dst = FFmpegVideo(dst, 'w+', fps=15)
    
    mask = RoiMask(src.height, src.width, boxes)
    fgbg = [cv2.createBackgroundSubtractorMOG2() for b in mask.bbox]

    def score(frame):  # motion_score_vector
        for j, (x, y, w, h) in enumerate(mask.bbox):
            motion = fgbg[j].apply(frame[y : y + h, x : x + w])
            motion_eroded = cv2.erode(motion, np.ones((2, 2), dtype = np.uint8))
            motion_eroded[mask.roi_mask[j]] = 0.0
            motion_eroded = motion_eroded > 254
            yield float(np.count_nonzero(motion_eroded)) / mask.polygon_area[j]


    last_written = float('-inf')
    start = datetime.now()
    
    framestream = buffer_half(((x, list(score(x))) for x in src), src.fps)
    
    for n, ((frame, _), buf) in enumerate(framestream):
        signal = np.array([b[1] for b in buf], dtype=np.float32)
        speed = (fast + (slow - fast) * np.clip(signal.max(axis=0) / level, 0.0, 1.0)).min()

        if n % 10000 == 0:
            print('.', end='', flush=True)
            
        if n - last_written < speed:
            continue

        for x, y, w, h in mask.bbox:
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED)

        dst.write(frame)
        last_written = n
        
        #if chunk_frames_written == 0:
            #cv2.imwrite(chunk.file_path + '.jpg', frame[:, :, ::-1])
            #print(chunk.file_path + '.jpg')
            
    print(' took', datetime.now() - start)


@command()
def cli():
    """
    For each uik/camera with high turnout, merge and speedup video.
    """
    
    boxes = json.load(open('voteboxes.json'))
    
    cams = [x for x in turnout() if int(x.turnout[:-1]) >= 70]
    ncams = len(cams)
    uiks = set(x.uik for x in cams)
    
    temp = '/mnt/2018-4TB-2/data/2018-Санкт-Петербург/concat/%(tik)s/%(uik)s_%(cam)s.mp4'
    dest = '/mnt/2018-4TB-2/data/2018-Санкт-Петербург/speedup/%(tik)s/%(uik)s_%(cam)s.mp4'
    
    n = 0
    for tikdir in Path(root).iterdir():
        tik = re.search('СПБ-2018-ТИК-(\d+)-.*', tikdir.name)
        if not tik:
            continue
        tik = 'TIK-' + tik.group(1)
        echo(tik)
        for camdir in tikdir.iterdir():
            uik, cam = re.search('r78_u(\d+)_(.+)', camdir.name).groups()
            
            #if not uik == '231':
            if uik not in uiks:
                continue
            
            n += 1
            print('%(n)s of %(ncams)s. %(tik)s %(uik)s %(cam)s' % locals())
            
            urna = boxes.get(uik, {}).get(cam, {}).get('boxes')
            if not urna:
                print(' ...not marked')
                continue
            
            tmp = temp % locals()
            if not exists(tmp):
                if not exists(dirname(tmp)):
                    makedirs(dirname(tmp))
                merge(camdir.iterdir(), tmp)
            
            dst = dest % locals()
            if exists(dst):
                print('..skipped existing')
                continue
            if not exists(dirname(dst)):
                makedirs(dirname(dst))
            print('speedup: ..', end='')
            speedup(tmp, dst, urna)
            os.remove(tmp)
    return
            
            
def turnout():
    csv = csv.reader(open('good.csv'), delimiter=',')
    next(csv, None)  # skip the headers
    
    Row = namedtuple('row', 'tik, uik, voters, turnout, putin, stationary, cam, gaps')
    for row in csv:
        yield Row(*row)
    #return {Row(*x).uik: Row(*x) for x in csv}


if __name__ == '__main__':
    cli()
    
