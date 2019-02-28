#!/usr/bin/env python

import csv
import json
import os
import re
import sys

from collections import defaultdict, Mapping, namedtuple
from datetime import datetime
from os import makedirs
from os.path import dirname, basename, join, exists
from pathlib import Path
from subprocess import Popen, PIPE, DEVNULL, check_output as sh

import numpy as np
import click
import cv2

from click import Context, confirm, command, option, argument, progressbar as progress

from more_itertools import windowed
from merge import merge
from regions import regions


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
            cmd = 'ffprobe -v quiet -print_format json -show_streams ' + self.file_path
            params = json.loads(sh(cmd, shell=True).decode('utf-8'))['streams'][0]
            self.height, self.width = params['height'], params['width'], 
            self.fps_, self.num_frames = params['avg_frame_rate'], int(params['nb_frames'])
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
        polygons = [np.array([[round(x*width), round(y*height)] for x, y in box]) for box in boxes]
        
        #polygons = [(np.clip(np.array(box, dtype=np.float32).reshape((-1, 2)), 0.0, 1.0) * np.array([width, height], dtype=np.float32)).astype(np.int32) for box in boxes]
        
        # Polygon should be at least 1 px wide
        for box in polygons:
            if np.array_equal(box[0], box[1]):
                box[0][0] = box[0][0] - 1
                box[0][1] = box[0][1] - 1
                
        self.bbox = list(map(cv2.boundingRect, polygons))
        self.polygon_area = list(map(cv2.contourArea, polygons))
        #print(polygons, self.bbox, self.polygon_area)
        full_mask = np.zeros((height, width), dtype = np.uint8)
        cv2.fillPoly(full_mask, polygons, 1)
        
        #timestamp_box_height = 40
        #cv2.rectangle(full_mask, (0, 0), (width, timestamp_box_height), 1, cv2.FILLED)
        
        self.roi_mask = [full_mask[y : y + h, x : x + w] != 1 for x, y, w, h in self.bbox]

            
def speedup(src, dst, boxes):
    """
    Speedup source file, given the boxes coordinates to detect motion.
    """
    level = 0.005
    slow, fast = 2, 16
    src = FFmpegVideo(src, 'r')
    dst = FFmpegVideo(dst, 'w+', fps=15)
    
    mask = RoiMask(src.height, src.width, boxes)
    #print(boxes, mask.polygon_area)
    fgbg = [cv2.createBackgroundSubtractorMOG2() for b in mask.bbox]

    def score(frame):  # motion_score_vector
        _score = []
        for j, (x, y, w, h) in enumerate(mask.bbox):
            motion = fgbg[j].apply(frame[y : y + h, x : x + w])
            motion_eroded = cv2.erode(motion, np.ones((2, 2), dtype = np.uint8))
            motion_eroded[mask.roi_mask[j]] = 0.0
            motion_eroded = motion_eroded > 254
            _score.append(float(np.count_nonzero(motion_eroded)) / mask.polygon_area[j])
        return frame, _score

    screenshot = True
    last_written = float('-inf')
    start = datetime.now()
    
    for n, win in enumerate(windowed(map(score, src), 20)):
        scores = np.array([score for (frame, score) in win], dtype=np.float32)
        speed = (fast + (slow - fast) * np.clip(scores.max(axis=0) / level, 0.0, 1.0)).min()

        if n % 10000 == 0:
            print('.', end='', flush=True)
            
        if n - last_written < speed:
            continue

        frame, _score = win[10]
        for x, y, w, h in mask.bbox:
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED)

        dst.write(frame)
        last_written = n
        
        if screenshot:
            cv2.imwrite(dst.file_path + '.jpg', frame[:, :, ::-1])
            ###print(dst.file_path + '.jpg')
            screenshot = False
            #break
            
            
    print(' took', datetime.now() - start)


@command()
@argument('uiks', nargs=-1)
@option('--region', '-rn', default='78', prompt=True, help='Region number')
@option('--turnout_min', '-tumin', default=0)
@option('--turnout_max', '-tumax', default=100)
#@option('--timestart', '-ts', default='07-45')
#@option('--timeend', '-te', default='20-00')
@option('--force', '-f', is_flag=True, default=False)
def cli(uiks, turnout_min, turnout_max, region, force):
    """
    For each uik/camera with high turnout, merge and speedup video.
    """
    
    ##h, m = (int(x) for x in timestart.split('-'))
    #tstart = datetime(2018, 3, 18, 7, 45)
    
    ##h, m = (int(x) for x in timeend.split('-'))
    #tend = datetime(2018, 3, 18, 20, 0)
    
    voteboxes = json.load(open(regions[region]['box_file']))
    
    if uiks:
        ncams = sum(len(voteboxes[x]) for x in uiks)
    else:
        cams = [x for x in turnout_csv(region) if int(turnout_min) <= int(x.turnout[:-1]) <= int(turnout_max)]
        ncams = len(cams)
        uiks = set(x.uik for x in cams)
    
    n = 0
    for tikdir in sorted(Path(regions[region]['src_dir']).iterdir()):
        tik = re.search(regions[region]['tik_pattern'], tikdir.name)
        if not tik:
            continue
        tik = 'TIK-' + tik.group(1)
        #echo(tik)
        for camdir in sorted(tikdir.iterdir()):
            uik, cam = re.search(regions[region]['uik_pattern'], camdir.name).groups()
            
            #if not uik == '231':
            if uik not in uiks:
                continue
            
            n += 1
            print('%(n)s of %(ncams)s. %(tik)s %(uik)s %(cam)s' % locals())
            
            boxes = voteboxes.get(uik, {}).get(cam, {}).get('boxes')
            if not boxes:
                print(' ...not marked')
                continue
            
            if boxes.keys() == {'07-45'}:
                temp = regions[region]['tmp_dir'] + '/%(tik)s/%(uik)s_%(cam)s.mp4'
                dest = regions[region]['dst_dir'] + '/speedup/%(tik)s/%(uik)s_%(cam)s.mp4'
            else:
                temp = regions[region]['tmp_dir'] + '/%(tik)s/%(uik)s_%(cam)s_%(boxtstart)s.mp4'
                dest = regions[region]['dst_dir'] + '/speedup/%(tik)s/%(uik)s_%(cam)s_%(boxtstart)s.mp4'
            
            timerange = sorted(boxes.keys()) + ['20-00']
            for i in range(len(boxes)):
                boxtstart, boxtend = timerange[i], timerange[i+1]
                
                dst = dest % locals()
                if exists(dst) and not force:
                    print('..skipped existing')
                    continue
                
                tmp = temp % locals()
                if not exists(tmp):
                    if not exists(dirname(tmp)):
                        makedirs(dirname(tmp))
                        
                    h, m = (int(x) for x in boxtstart.split('-'))
                    tstart = datetime(2018, 3, 18, h, m)
                    
                    h, m = (int(x) for x in boxtend.split('-'))
                    tend = datetime(2018, 3, 18, h, m)
                    merge(camdir.iterdir(), tmp, tstart, tend)
                
                if not exists(dirname(dst)):
                    makedirs(dirname(dst))
                print('speedup: ..', end='', flush=True)
                speedup(tmp, dst, boxes[boxtstart])
                os.remove(tmp)
    return
            
            
def turnout_csv(region):
    if region == '47':
        data = csv.reader(open('47_good.csv'), delimiter=',')
        next(data, None)  # skip the headers

        Row = namedtuple('row', 'tik uik turnout stationary putin koib cam marked gaps')
        for row in data:
            yield Row(*row)
    else:
        data = csv.reader(open('good.csv'), delimiter=',')
        next(data, None)  # skip the headers

        Row = namedtuple('row', 'tik uik voters turnout putin stationary cam marked gaps mn_in mn_out koib')
        for row in data:
            yield Row(*row)
        #return {Row(*x).uik: Row(*x) for x in csv}


if __name__ == '__main__':
    cli()
    
