#!/usr/bin/env python3
import csv
import json
import re
    
from datetime import datetime
from collections import defaultdict, Mapping, namedtuple
from itertools import chain
from os.path import exists
from pathlib import Path
from subprocess import Popen, PIPE

import click


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
            dst = target.get(key, getattr(target, 'default_factory', dict)())
            target[key] = update(dst, value)
        else:
            target[key] = source[key]
    return target

       
@click.command()
@click.option('--file', '-f', help='JSON file to write the report.',
              type=click.Path(dir_okay=False, writable=True))
@click.option('--quiet', '-q', is_flag=True, default=False, help='Do not print progress to stdout')
@click.option('--skip-good/--no-skip-good', default=True)
@click.option('--skip-invalid/--no-skip-invalid', default=True, help='Skip invalid files')
@click.option('--skip-dur-err/--no-skip-dur-err', default=True, help='Skip files with duration error')
@click.option('--skip-diff-err/--no-skip-diff-err', default=True, help='Skip files with max diff error')
def cli(file=None, quiet=False, 
        skip_good=True, skip_diff_err=True, skip_dur_err=True, skip_invalid=True):
    """
    Generate json gap report for tik/uik_cam video files starting from root dir.
    """
    global QUIET
    QUIET = quiet
    turnreport = turnout()
    votes = voters()
    statsreport = stats()
    putin = putindata()
    
    def st(uik): 
        koib = ''
        mnin, mnout = '?', '?'
        result = statsreport.get(uik, [])
        if result:
            if result[6] == '1':
                koib = 'КОИБ'
            elif result[6] == '2':
                koib = 'КЭГ'
            mnin, mnout = result[4], result[5]
        #return votes[uik].mnin, mnin, votes[uik].mnout, mnout, koib
        return mnin, mnout, koib
        #return ('?', '?', '?')
    
    boxes = json.load(open('voteboxes.json'))
    report = {} 
    for tik, uiks in json.load(open('rr.json')).items():
        for uik in uiks:
            report[uik] = uiks[uik]
        
    head = 'tik uik voters turnout putin stationary cam marked gaps mn_in mn_out koib'.split()
    
    allrows = []
    absent = csv.writer(open('absent.csv', 'w+'))
    absent.writerow(head)
    
    for uik in sorted(set(turnreport) - set(report), key=lambda x: -float(turnreport[x].turnout[:-1])):
        row = turnreport[uik] + (putin[uik], votes[uik].stationary, '', '', 'missing') + st(uik)
        absent.writerow(row)
        allrows.append(row)
        
    goodcsv = csv.writer(open('good.csv', 'w+'))
    goodcsv.writerow(head)
    goodmarkedcsv = csv.writer(open('good_marked.csv', 'w+'))
    goodmarkedcsv.writerow(head)
    badcsv = csv.writer(open('bad.csv', 'w+'))
    badcsv.writerow(head)
    allcsv = csv.writer(open('all.csv', 'w+'))
    allcsv.writerow(head)
    for uik in sorted(report, key=lambda x: -float(turnreport[x].turnout[:-1])):
        good = True
        bad = False
        verybad = False
        #tik = report[uik].pop('tik')
        gaps = defaultdict(list)
        for cam, data in report[uik].items():
            #print(tik, uik, cam, data)
            data.pop('status')
            for file in data:
                begin, end = re.search(r'(\d{10})_(\d{10})?', str(file)).groups()
                begin = datetime.utcfromtimestamp(float(begin) + 3 * 3600)  # MSK
                if 8 < begin.hour < 20:
                    diff = data[file].get('maxdiff_error')
                    if diff:
                        gaps[cam].append(int(diff))
                    
                    dur = data[file].get('duration_error')
                    if dur:
                        gaps[cam].append(int(900 - dur))
                        
                    if 'no_timestamps' in data[file]:
                        gaps[cam].append('X')
        uikgaps = list(chain(*gaps.values()))
        if not uikgaps:
            for cam in report[uik]:
                marked = 'yes' if boxes[uik][cam]['boxes'] else ''
                row = turnreport[uik] + (putin[uik], votes[uik].stationary, cam, marked, '') + st(uik)
                goodcsv.writerow(row)
                if marked:
                    goodmarkedcsv.writerow(row)
                allrows.append(row)
        else:
            for cam in report[uik]:
                marked = 'yes' if boxes[uik][cam]['boxes'] else ''
                gap = ', '.join(map(str, gaps[cam]))
                row = turnreport[uik] + (putin[uik], votes[uik].stationary, cam, marked, gap) + st(uik)
                badcsv.writerow(row)
                allrows.append(row)
                
    for row in sorted(allrows, key=lambda x: -float(x[3][:-1])):
        allcsv.writerow(row)
    
    
def turnout():
    data = csv.reader(open('turnout.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'tik, uik, voters, turnout')
    new = lambda x: Row(*x[:len(Row._fields)])
    return {new(x).uik: new(x) for x in data}

def stats():
    data = csv.reader(open('stats.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'tik, raion, uik, voters, mn_in, mn_out, koib')
    new = lambda x: Row(*x[:len(Row._fields)])
    #for row in data:
        #try:
            #Row(*row[:len(Row._fields)])
        #except:
            #print(row)
            #raise
    return {new(x).uik: new(x) for x in data}


def voters():
    data = csv.reader(open('spb_2018.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'id, reg, tik, uik, totjan, mnin, mnout, excl, totmarch, t10, t12, t15, t18, totvoters, got, pre, given, out, dicard, mob, stationary')
    new = lambda x: Row(*x[:len(Row._fields)])
    return {new(x).uik: new(x) for x in data}
    
def putindata():
    data = csv.reader(open('spb_2018_calc_edt_4roman.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'tik, uik, vidano,  turnout, putin')
    return {Row(*x).uik: f"{round(float(Row(*x).putin.replace(',', '.')) * 100)}%" for x in data}

if __name__ == '__main__':
    cli()
