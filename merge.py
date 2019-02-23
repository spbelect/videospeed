#!/usr/bin/env python
import json
import os
import re
import subprocess

from datetime import datetime
from collections import defaultdict, Mapping, namedtuple
from os.path import exists, dirname
from os import makedirs
from pathlib import Path
from subprocess import Popen, PIPE, DEVNULL, check_output as sh

from click import Context, confirm, command, option, group, argument, progressbar as progress


QUIET = False

echo = lambda *a, **kw: not QUIET and print(*a, **kw, flush=True)

printchar = lambda *a, **kw: echo(*a, end='', **kw)

Context.get_usage = Context.get_help  # show full help on error


tstamp = lambda x: float(re.search(r'(\d{10})_(\d{10})?', str(x.stem)).groups()[0])


def merge(files, dst, timestart=None, timeend=None):
    input = []
    findhash = None
    outpoint = 0
    Frame = namedtuple('Frame', 'stream, dts, pts, duration, size, hash')
    
    _files = []
    for file in sorted(files, key=tstamp):
        begin = datetime.utcfromtimestamp(tstamp(file) + 3 * 3600)  # MSK
        if timestart and begin < timestart:
            continue
        if timeend and begin > timeend:
            continue
        _files.append(file)

    ##with progress(, label='Merging') as bar:
    echo('Merging ..', end='')
    # TODO: do not reverse, calculate inpoint instead of outpoint.
    for file in reversed(_files):
        cmd = 'ffmpeg -i %s -an -f framemd5 -c copy -' % file
        data = sh(cmd, stderr=DEVNULL, shell=True).split(b'\n')
        
        timebase = [x for x in data if x.startswith(b'#tb')][0]
        tb_num, tb_den = list(map(int, timebase.split()[-1].split(b'/')))
        
        frames = [x.replace(b',', b'').split() for x in data if x and not x.startswith(b'#')]
        
        for line in reversed(frames[1:]):
            frame = Frame(*line)
            if frame.hash == findhash:
                outpoint = float(frame.pts) * tb_num / tb_den
                break
            
        if outpoint:
            input.append('file %s\noutpoint %s' % (file, outpoint))
        else:
            input.append('file %s' % file)
        #print(int(outpoint), end=';', flush=True)
        echo('.', end='')
        findhash = Frame(*frames[0]).hash
    

    cmd = 'ffmpeg -nostats -hide_banner -avoid_negative_ts make_zero -fflags +genpts -f concat -safe 0' \
        ' -protocol_whitelist file,pipe -i - -c copy -flags +global_header -movflags +faststart -y '
    #print(cmd + dst)
    pipe = Popen(cmd + dst, stdin=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = pipe.communicate('\n'.join(reversed(input)).encode('utf8'))
    if pipe.returncode != 0:
        raise Exception(stderr.decode('utf8'))
    print()


@group()
def cli():
    """
    """
    pass
    
    
@cli.command()
@argument('uiks', nargs=-1, required=True)
@option('--timestart', '-ts', default='07-45')
@option('--timeend', '-te', default='20-00')
def uiks(uiks, timestart, timeend):
    """
    For each uik/camera, merge video.
    """
    
    root = '/mnt/ftp/2018-Санкт-Петербург/'
    boxes = json.load(open('voteboxes.json'))
    
    ncams = sum(len(boxes[x]) for x in uiks)
    
    temp = '/mnt/2018-4TB-2/data/2018-Санкт-Петербург/concat/%(tik)s/%(uik)s_%(cam)s.mp4'
    
    h, m = (int(x) for x in timestart.split('-'))
    tstart = datetime(2018, 3, 18, h, m)
    
    h, m = (int(x) for x in timeend.split('-'))
    tend = datetime(2018, 3, 18, h, m)
    
    n = 0
    for tikdir in Path(root).iterdir():
        tik = re.search('СПБ-2018-ТИК-(\d+)-.*', tikdir.name)
        if not tik:
            continue
        tik = 'TIK-' + tik.group(1)
        #echo(tik)
        for camdir in tikdir.iterdir():
            uik, cam = re.search('r78_u(\d+)_(.+)', camdir.name).groups()
            
            #if not uik == '231':
            if uik not in uiks:
                continue
            
            n += 1
            print('%(n)s of %(ncams)s. %(tik)s %(uik)s %(cam)s' % locals())
            
            #urna = boxes.get(uik, {}).get(cam, {}).get('boxes')
            #if not urna:
                #print(' ...not marked')
                #continue
            
            tmp = temp % locals()
            if exists(tmp):
                print(' ...skip existing')
                continue
                
            if not exists(dirname(tmp)):
                makedirs(dirname(tmp))
            merge(camdir.iterdir(), tmp, tstart, tend)
       

if __name__ == '__main__':
    cli()
