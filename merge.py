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

from click import Context, confirm, command, option, progressbar as progress


QUIET = False

echo = lambda *a, **kw: not QUIET and print(*a, **kw)

printchar = lambda *a, **kw: echo(*a, end='', flush=True, **kw)

Context.get_usage = Context.get_help  # show full help on error


def merge(files, dst):
    input = []
    findhash = None
    outpoint = 0
    Frame = namedtuple('Frame', 'stream, dts, pts, duration, size, hash')

    with progress(reversed(sorted(list(files))), label='Merging') as bar:
        for file in bar:
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
            findhash = Frame(*frames[0]).hash
        

    cmd = 'ffmpeg -nostats -hide_banner -avoid_negative_ts make_zero -fflags +genpts -f concat -safe 0' \
          ' -protocol_whitelist file,pipe -i - -c copy -flags +global_header -movflags +faststart -y '
    #print(cmd + dst)
    pipe = Popen(cmd + dst, stdin=PIPE, stderr=DEVNULL, shell=True)
    pipe.communicate('\n'.join(reversed(input)).encode('utf8'))

    #return produced_files[0] if len(produced_files) == 1 else None

       

#if __name__ == '__main__':
    #cli()
