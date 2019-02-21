#!/usr/bin/env python
import re
import json

from datetime import datetime
from collections import defaultdict, Mapping
from os.path import exists
from pathlib import Path
from subprocess import Popen, PIPE

import click
import environ

from regions import regions


env = environ.Env()

SRC_DIR = environ.Path(__file__) - 1  # ./

env.read_env(SRC_DIR('env-local'))  # overrides env-default
env.read_env(SRC_DIR('env-default'))  


QUIET = False

echo = lambda *a, **kw: not QUIET and print(*a, **kw)

printchar = lambda *a, **kw: echo(*a, end='', flush=True, **kw)

click.Context.get_usage = click.Context.get_help  # show full help on error


def update(target, source):
    """
    Recursively update dictionary.
    """
    for key, value in source.items():
        if isinstance(value, Mapping):
            # If target is defaultdict, call its `default_factory()`
            dst = target.get(key) or getattr(target, 'default_factory', dict)()
            target[key] = update(dst, value)
        else:
            target[key] = source[key]
    return target

### spb

def timestamps(file):
    """ Iterate over Presentation Timestamps of packets. """
    cmd = 'ffprobe -loglevel fatal -hide_banner -of compact' \
          ' -select_streams v:0 -show_entries packet=pts_time '
    for line in Popen(cmd + str(file), shell=True, stdout=PIPE).stdout:
        if line.strip():
            yield float(re.search(b'packet\|pts_time=(\d+.\d+)', line).group(1))
       
       
@click.command()
@click.option('--root', '-rd', type=click.Path(exists=True), envvar='ROOT', help='Root dir of region')
@click.option('--file', '-f', help='JSON file to write the report.',
              type=click.Path(dir_okay=False, writable=True), envvar='GAPFILE')
@click.option('--region', '-rn', default='78', prompt=True, help='Region number')
@click.option('--quiet', '-q', is_flag=True, default=False, help='Do not print progress to stdout')
@click.option('--skip-good/--no-skip-good', default=True)
@click.option('--skip-invalid/--no-skip-invalid', default=True, help='Skip invalid files')
@click.option('--skip-dur-err/--no-skip-dur-err', default=True, help='Skip files with duration error')
@click.option('--skip-diff-err/--no-skip-diff-err', default=True, help='Skip files with max diff error')
def cli(region, root=None, duration=900, maxdiff=2, file=None, quiet=False, 
        skip_good=True, skip_diff_err=True, skip_dur_err=True, skip_invalid=True):
    """
    Generate json gap report for tik/uik_cam video files starting from root dir.
    """
    global QUIET
    QUIET = quiet
    if not root:
        root = regions[region]['root_dir']
        
    report = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    if file and exists(file):
        data = open(file).read()
        if data:
            try:
                data = json.loads(data)
            except:
                click.confirm('File is not a valid json and will be overwritten. Continue?', abort=True)
            else:
                echo('file ok')
                update(report, data)
        
    try:
        gapreport(root, region, report, duration, maxdiff, skip_good, skip_diff_err, skip_dur_err, skip_invalid)
    finally:
        report = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        if file:
            open(file, 'wb+').write(report.encode('utf8'))
        else:
            print(report)
        
    
def gapreport(root, region, report, duration=900, maxdiff=2, 
              skip_good=True, skip_diff_err=True, skip_dur_err=True, skip_invalid=True):
    boxes = json.load(open('boxes.json'))
    voteboxes = defaultdict(lambda: defaultdict(dict))
    update(voteboxes, json.load(open('voteboxes.json')))
    #root = Path(root)
    #root.rename(root.parent / '2018-Spb')
    #return
    for tikdir in sorted(Path(root).iterdir()):
        tik = re.search(regions[region]['tik_pattern'], tikdir.name)
        if not tik:
            continue
        
        #name = tikdir.name.replace('СПБ', 'spb').replace('ТИК', 'TIK').replace('УИК', 'UIK')
        #print(name)
        #tikdir.rename(tikdir.parent / name)
        #continue
        tik = 'TIK-' + tik.group(1)
        echo(tik)
        for camdir in sorted(tikdir.iterdir()):
            uik, cam = re.search(regions[region]['uik_pattern'], camdir.name).groups()
            echo(uik, cam, ' ', end='')
            camdata = report[tik][uik][cam]
            
            interval = set('08:00 08:15 08:30 08:45 09:00 09:15 09:30 09:45 10:00 10:15 10:30 '
                '10:45 11:00 11:15 11:30 11:45 12:00 12:15 12:30 12:45 13:00 13:15 13:30 13:45 '
                '14:00 14:15 14:30 14:45 15:00 15:15 15:30 15:45 16:00 16:15 16:30 16:45 17:00 '
                '17:15 17:30 17:45 18:00 18:15 18:30 18:45 19:00 19:15 19:30 19:45'.split())
            for file in sorted(camdir.iterdir()):
                #camid = str(file.stem).split('_')[-1]
                #break
                begin, end = re.search(r'(\d{10})_(\d{10})?', str(file.stem)).groups()
                begin = datetime.utcfromtimestamp(float(begin) + 3 * 3600)  # MSK
                #prefix = begin.strftime('%H-%M_')
                #if not file.name.startswith(prefix):
                    #print(file)
                    #file.rename(file.parent / (prefix + file.name.replace('segment_', '')))
                    
                #break
                interval -= {begin.strftime('%H:%M'),}
            #break
            #continue
            if interval:
                printchar('M ', ' '.join(sorted(interval)))
                camdata['missing'] = ' '.join(sorted(interval))
            #if not boxes.get(camid, {}).get('boxes'):
                #print(tik, uik, cam, camid, dict(report[tik][uik][cam]))
            
            #voteboxes[region][uik][cam] = {'id': camid, 'boxes': boxes.get(camid, {}).get('boxes', [])}
            #continue

            if camdata.get('status'):
                # Camera already checked.
                if len(camdata) == 1:
                    if skip_good:
                        echo('skip good')
                        continue
                    for file in sorted(camdir.iterdir()):
                        check_one(file, camdata, duration, maxdiff)
                else:
                    # Some camera file has errors.
                    for file in sorted(camdir.iterdir()):
                        if file.name in camdata:
                            # File has errors.
                            if skip_invalid and skip_diff_err and skip_dur_err:
                                continue
                            if not skip_dur_err and 'duration_error' in camdata[file.name]:
                                check_one(file, camdata, duration, maxdiff)
                            elif not skip_diff_err and 'maxdiff_error' in camdata[file.name]:
                                check_one(file, camdata, duration, maxdiff)
                            elif not skip_invalid and 'no_timestamps' in camdata[file.name]:
                                check_one(file, camdata, duration, maxdiff)
                            else:
                                check_one(file, camdata, duration, maxdiff)
                        else:
                            # File good.
                            if skip_good:
                                continue
                            check_one(file, camdata, duration, maxdiff)
                        
            else:
                # Camera not checked yet.
                for file in sorted(camdir.iterdir()):
                    check_one(file, camdata, duration, maxdiff)
            camdata['status'] = 'checked'
            echo()
    #xr = json.dumps(voteboxes, indent=2, sort_keys=True, ensure_ascii=False)
    #open('voteboxes.json', 'wb+').write(xr.encode('utf8'))


def check_one(file, camdata, duration=900, maxdiff=2):
    if not file.suffix == '.flv':
        return
    printchar('.')
    begin, end = re.search(r'(\d{10})_(\d{10})?', str(file)).groups()
    begin = datetime.utcfromtimestamp(float(begin) + 3 * 3600)  # MSK
    
    ts = list(timestamps(file))
    if not ts:
        camdata[file.name]['no_timestamps'] = file.stat().st_size
        camdata[file.name]['time'] = begin.strftime('%H:%M')
        printchar('-', begin.strftime('%H:%M'))
        return
    if ts[-1] < duration:
        camdata[file.name]['duration_error'] = ts[-1]
        camdata[file.name]['time'] = begin.strftime('%H:%M')
        printchar('x', begin.strftime('%H:%M'), int(ts[-1]))
    diffs = [y - x for x, y in zip(ts[:-1], ts[1:])]
    diff = max(diffs)
    if diff > maxdiff:
        m, s = divmod(int(ts[diffs.index(diff)]), 60)
        camdata[file.name]['maxdiff_error'] = diff
        camdata[file.name]['time'] = begin.strftime('%H:%M:00') + ' + 00:%02d:%02d' % (m, s)
        printchar('X', begin.strftime('%H:%M:00') + ' + 00:%02d:%02d' % (m, s), int(diff))
    

if __name__ == '__main__':
    cli()
