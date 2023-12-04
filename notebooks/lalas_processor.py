import re

def get_longest_anti(s):
    adjacent_anti = re.compile('[AFC]+')
    matches = adjacent_anti.findall(s)
    longest = 0
    results = ''
    for m in matches:
        if len(m) > longest:
            longest = len(m)
            results = m
    return longest

def get_longest_helix(s, direction):
    clockwise = re.compile(f'\{direction}+')
    branches = re.compile('\(.+?\)')
    matches = clockwise.findall(s)

    longest = 0
    results = ''
    for m in matches:
        m = branches.sub('', m)
        if len(m) > longest:
            longest = len(m)
            results = m
    return longest


def get_longest_L(s):
    adjacent_ls = re.compile('L+')
    matches = adjacent_ls.findall(s)
    
    longest = 0
    results = ''
    for m in matches:
        if len(m) > longest:
            longest = len(m)
            results = m
    return longest

def get_longest_A(s):
    adjacent_as = re.compile('[TA]+\(.+?\)[TA]+')
    branches = re.compile('\(.+?\)')
    matches = adjacent_as.findall(s)

    longest = 0
    results = ''
    for m in matches:
        m = branches.sub('', m)
        if len(m) > longest:
            longest = len(m)
            results = m
    return longest