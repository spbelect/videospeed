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

from click import Context, confirm, command, option, argument

from regions import regions


QUIET = False

echo = lambda *a, **kw: not QUIET and print(*a, **kw)

printchar = lambda *a, **kw: echo(*a, end='', flush=True, **kw)

Context.get_usage = Context.get_help  # show full help on error


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

       
@command()
@option('--region', '-rn', default='78', prompt=True, help='Region number')
def cli(region):
    """
    Generate csv files.
    """
    turnreport = turnout()
    votes = voters()
    statsreport = stats()
    putin = putindata()
    
    if region == '47':
        regdata = {x.uik: x for x in lendata()}
    else:
        regdata = {x.uik: x for x in spbdata()}
        
    #def st(uik):
        #result = statsreport.get(uik, [])
        #if not result:
            #return '?', '?', ''
        
        #koib = ''
        #if result.koib == '1':
            #koib = 'КОИБ'
        #elif result.koib == '2':
            #koib = 'КЭГ'
        #return result.mnin, result.mnout, koib
    
    boxes = json.load(open('voteboxes.json'))[region]
    gapreport = {} 
    for tik, uiks in json.load(open(regions[region]['gap_file'])).items():
        for uik in uiks:
            gapreport[uik] = uiks[uik]
        
    #head = 'tik uik voters turnout putin stationary cam marked gaps mn_in mn_out koib'.split()
    head = 'tik uik turnout stationary putin koib cam marked gaps'.split()
    
    allrows = []
    absent = csv.writer(open(f'{region}_absent.csv', 'w+'))
    absent.writerow(head)
    
    for uik in sorted(set(regdata) - set(gapreport), key=lambda x: -float(regdata[x].turnout[:-1])):
        row = regdata[uik] + ('', '', 'missing')
        absent.writerow(row)
        allrows.append(row)
        
    goodcsv = csv.writer(open(f'{region}_good.csv', 'w+'))
    goodcsv.writerow(head)
    goodmarkedcsv = csv.writer(open(f'{region}_good_marked.csv', 'w+'))
    goodmarkedcsv.writerow(head)
    badcsv = csv.writer(open(f'{region}_bad.csv', 'w+'))
    badcsv.writerow(head)
    allcsv = csv.writer(open(f'{region}_all.csv', 'w+'))
    allcsv.writerow(head)
    for uik in sorted(gapreport, key=lambda x: -float(regdata[x].turnout[:-1])):
        
        bad = False
        #verybad = False
        #tik = gapreport[uik].pop('tik')
        gaps = defaultdict(lambda: defaultdict(list))
        #gapfiles = set()
        for cam, data in gapreport[uik].items():
            #print(tik, uik, cam, data)
            data.pop('status')
            for file in data:
                begin, end = re.search(r'(\d{10})_(\d{10})?', str(file)).groups()
                begin = datetime.utcfromtimestamp(float(begin) + 3 * 3600)  # MSK
                if 8 <= begin.hour < 20:
                    diff = data[file].get('maxdiff_error')
                    if diff:
                        gaps[cam][file[:5]].append(int(diff))
                    
                    dur = data[file].get('duration_error')
                    if dur:
                        gaps[cam][file[:5]].append(int(900 - dur))
                        
                    if 'no_timestamps' in data[file]:
                        gaps[cam][file[:5]].append('X')
                    
                    
        good = True
        if region == '47':
            for cam in gaps.values():
                for file in cam: 
                    if not re.match('(08-45|15-00).*', file):
                        good = False
                        break
        else:
            good = not gaps
             
        #uikgaps = list(chain(*gaps.values()))
        #import ipdb; ipdb.sset_trace()
        if good:
            for cam in gapreport[uik]:
                marked = 'yes' if boxes[uik][cam]['boxes'] else ''
                gap = ', '.join(map(str, chain(*gaps[cam].values())))
                row = regdata[uik] + (cam, marked, gap)
                goodcsv.writerow(row)
                if marked:
                    goodmarkedcsv.writerow(row)
                allrows.append(row)
        else:
            for cam in gapreport[uik]:
                marked = 'yes' if boxes[uik][cam]['boxes'] else ''
                gap = ', '.join(map(str, chain(*gaps[cam].values())))
                row = regdata[uik] + (cam, marked, gap)
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
    
    Row = namedtuple('row', 'id, reg, tik, uik, totjan, mnin, mnout, excl, totmarch, t10, t12, t15, t18, totvoters, got, pre, given, out, discard, mob, stationary')
    new = lambda x: Row(*x[:len(Row._fields)])
    return {new(x).uik: new(x) for x in data}
    
def putindata():
    data = csv.reader(open('spb_2018_calc_edt_4roman.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'tik, uik, vidano,  turnout, putin')
    return {Row(*x).uik: f"{round(float(Row(*x).putin.replace(',', '.')) * 100)}%" for x in data}


def lendata():
    
    params = csv.reader(open('lo-params.csv'), delimiter=',')
    next(params, None)  # skip the headers
    
    Params = namedtuple('Params', 'region tik uik koib')
    new = lambda x: Params(*x[:4])
    params = {new(x).uik: new(x) for x in params}
    
    data = csv.reader(open('lenobl2018.csv'), delimiter=',')
    next(data, None)  # skip the headers
    
    Row = namedtuple('row', 'pu turnout tik uik totjan t10 t12 t15 t18 totvoters got pre given out discard mob stationary')
    
    Result = namedtuple('Result', 'tik uik turnout stationary putin koib')
    
    for row in data:
        x = Row(*row[:len(Row._fields)])
        pu = f'{round(float(x.pu) * 100)}%'
        turnout = f'{round(float(x.turnout) * 100)}%'
        
        par = params.get(x.uik)
        koib = ''
        if par and par.koib == '1':
            koib = 'КОИБ'
        elif par and par.koib == '2':
            koib = 'КЭГ'
            
        yield Result(x.tik, x.uik, turnout, x.stationary, pu, koib)
        #{Row(*x).uik: f"{)}%" for x in data}
    
if __name__ == '__main__':
    cli()
