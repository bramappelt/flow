import os
import re
import itertools

import pandas as pd


class OutputParser:
    data = {'Nod_inf.out': (r'\sTime:\s+([0-9]+\.[0-9]+)\n',
                            r'end\n'),
            'Balance.out': (r'\sTime\s+\[T\]\s+([0-9]+\.[0-9]+)\n',
                            r'\s?\n'),
            'T_Level.out': (r'\n', r'end\n'),
            'Run_inf.out': (r'\n', r'end\n'),
            'Profile.out': (r'\n', r'end\n')}

    upattern = r'\s*(\[.+\])+\n'

    def __init__(self, directory):
        self.dir = directory

    def df_locator(self, filename, spattern=None, epattern=None,
                   upattern=None, firstmatch=False):
        filepath = os.path.join(self.dir, filename)
        dflt_spattern, dflt_epattern = self.data.get(filename, (None, None))
        spattern = re.compile(spattern or dflt_spattern)
        epattern = re.compile(epattern or dflt_epattern)
        upattern = re.compile(upattern or self.upattern)

        startpoints, endpoints, units = [], [], []
        with open(filepath, 'r') as fr:
            for idx, line in enumerate(fr.readlines()):
                smatch = re.match(spattern, line)
                if smatch:
                    try:
                        key = float(smatch.group(1))
                    except IndexError:
                        key = idx
                    startpoints.append((idx, key))
                elif re.match(upattern, line):
                    units.append(idx)
                else:
                    if re.match(epattern, line) and startpoints:
                        endpoints.append((idx))
        if firstmatch:
            startpoints = startpoints[:1]

        return startpoints, endpoints, units, idx

    def df_parse(self, filename, sidx, eidx, uidx, flen, include_sidx, **kwargs):
        filepath = os.path.join(self.dir, filename)
        sidx, key = sidx
        sep = r'\s+'
        skipfooter = flen - eidx + 1

        skiprows = list(range(sidx + int(not include_sidx)))
        if uidx is not None:
            skiprows = skiprows + [uidx]

        df = pd.read_csv(filepath, skiprows=skiprows, skipfooter=skipfooter,
                         sep=sep, engine='python', **kwargs)
        df.key = key
        return df

    def df_parser(self, filename, spattern=None, epattern=None,
                  upattern=None, firstmatch=False, include_sidx=False,
                  engine='python', **kwargs):
        dfs = []
        sidc, eidc, uidc, flen = self.df_locator(filename, spattern,
                                                 epattern, upattern,
                                                 firstmatch=firstmatch)
        for s, e, u in itertools.zip_longest(sidc, eidc, uidc):
            df = self.df_parse(filename, s, e, u, flen,
                               include_sidx=include_sidx, **kwargs)
            dfs.append((df.key, df))

        if len(dfs) > 1:
            dfs = dict(((k, v) for k, v in dfs))
        else:
            dfs = df
        return dfs


def parse_obs_node(sourcedir):
    filepath = os.path.join(sourcedir, 'Obs_Node.out')
    mainheading, subheading = None, None
    with open(filepath, 'r') as fr:
        for idx, line in enumerate(fr.readlines()):
            if 'Node(' in line:
                line = re.split('\s{5,}', line)
                line = [l.replace('\n', '') for l in line]
                mainheading = (idx, line)
            elif 'time' in line:
                subheading = (idx, line.split())
    if mainheading is not None:
        columns = [(i, j) for i, j in zip(mainheading[1:], subheading[1:])]
        print(columns)
        df = pd.read_csv(filepath, sep=r'\s+', skipfooter=2, skiprows=subheading[0], header=0)
        df.columns = columns
        return df



def get_all_hydrus_frames(sourcedir):
    data = {}
    filenames = ['Nod_inf.out', 'T_Level.out', 'Run_inf.out', 'Profile.out',
                 'I_check.out', 'Obs_Node.out']
    OP = OutputParser(sourcedir)
    for idx, filename in enumerate(filenames):
        if idx == 0:
            data[filename] = OP.df_parser(filename)
        elif 1 <= idx < 4:
            data[filename] = OP.df_parser(filename, firstmatch=True)
        elif 4 <= idx < 5:
            spatt = r'Nodal point information'
            epatt = r'end\n'
            s4, e4, u4, flen4 = OP.df_locator(filename, spattern=spatt, epattern=epatt)
            df4 = OP.df_parse(filename, s4[0], e4[0], None, flen=flen4, include_sidx=False)
            data[filename + '1'] = df4 

            spatt = r'\s\n'
            epatt = r'end\n'
            s5, e5, u5, flen5 = OP.df_locator(filename, spattern=spatt, epattern=epatt)
            df5 = OP.df_parse(filename, s5[-1], e5[-1], None, flen=flen5, include_sidx=False)
            cols = ['theta', 'h', 'log_h', 'C', 'K', 'log_K', 'S', 'Kv']
            df5.columns = cols
            data[filename + '2'] = df5
        else:
            data[filename] = 'NOT AVAILABLE'
    return data


if __name__ == '__main__':
    sourcedir = "C:\\Users\\bramb\\Documents\\thesis\\compare\\Standard"
    # sourcedir = "C:\\Users\\bramb\\Documents\\thesis\\compare\\spacings\\0.75"
    filename = ['Nod_inf.out', 'T_Level.out', 'Run_inf.out', 'Profile.out',
                'I_check.out', 'Obs_Node.out']
    OP = OutputParser(sourcedir)

    s0, e0, u0, flen0 = OP.df_locator(filename[0])
    df0 = OP.df_parser(filename[0])

    s1, e1, u1, flen1 = OP.df_locator(filename[1], firstmatch=True)
    df1 = OP.df_parser(filename[1], firstmatch=True)

    s2, e2, u2, flen2 = OP.df_locator(filename[2], firstmatch=True)
    df2 = OP.df_parser(filename[2], firstmatch=True)

    s3, e3, u3, flen3 = OP.df_locator(filename[3], firstmatch=True)
    df3 = OP.df_parser(filename[3], firstmatch=True)

    spatt = r'Nodal point information'
    epatt = r'end\n'
    s4, e4, u4, flen4 = OP.df_locator(filename[4], spattern=spatt,
                                      epattern=epatt)
    df4 = OP.df_parse(filename[4], s4[0], e4[0], None, flen=flen4, include_sidx=False)

    spatt = r'\s\n'
    epatt = r'end\n'
    s5, e5, u5, flen5 = OP.df_locator(filename[4], spattern=spatt,
                                      epattern=epatt)
    df5 = OP.df_parse(filename[4], s5[-1], e5[-1], None, flen=flen5, include_sidx=False)
    cols = ['theta', 'h', 'log_h', 'C', 'K', 'log_K', 'S', 'Kv']
    df5.columns = cols

    HYDRUS_DFDATA = get_all_hydrus_frames(sourcedir)
