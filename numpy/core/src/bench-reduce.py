"""
%prog [OPTIONS] PATH [SECTION | SHAPE TRANSPOSE INDEX]
       %prog [OPTIONS] plot < INPUT

Benchmark numpy.sum for different size and stride arrays.
The `PATH` determines a new sys.path entry where an alternative
version of numpy can be found. E.g. ``build/lib.linux-i686-2.6``.

"""
import sys, os, glob, subprocess, timeit, time, itertools
import optparse, random

#------------------------------------------------------------------------------
# Benchmarks
#------------------------------------------------------------------------------

SUITE = """
# big-2d
2,8000     0,1   0
4,8000     0,1   0
8,8000     0,1   0
32,8000    0,1   0
256,8000   0,1   0
8000,2     1,0   0
8000,4     1,0   0
8000,8     1,0   0
8000,32    1,0   0
8000,256   1,0   0
8000,2     0,1   0
8000,4     0,1   0
8000,8     0,1   0
8000,32    0,1   0
8000,256   0,1   0

# medium-2d
2,800      0,1   0
4,800      0,1   0
8,800      0,1   0
32,800     0,1   0
256,800    0,1   0
800,2      1,0   0
800,4      1,0   0
800,8      1,0   0
800,32     1,0   0
800,256    1,0   0
800,2      0,1   0
800,4      0,1   0
800,8      0,1   0
800,32     0,1   0
800,256    0,1   0

# small-2d
2,80       0,1   0
4,80       0,1   0
8,80       0,1   0
32,80      0,1   0
256,80     0,1   0
80,2       1,0   0
80,4       1,0   0
80,8       1,0   0
80,32      1,0   0
80,256     1,0   0
80,2       0,1   0
80,4       0,1   0
80,8       0,1   0
80,32      0,1   0
80,256     0,1   0

# small-2d-other-axis
80,2       0,1   1
80,4       0,1   1
80,8       0,1   1
80,32      0,1   1
80,256     0,1   1

# tiny-2d
2,10       0,1   0
4,10       0,1   0
8,10       0,1   0
32,10      0,1   0
256,10     0,1   0
10,2       1,0   0
10,4       1,0   0
10,8       1,0   0
10,32      1,0   0
10,256     1,0   0
10,2       0,1   0
10,4       0,1   0
10,8       0,1   0
10,32      0,1   0
10,256     0,1   0

# tiny-2d-other-axis
10,2       0,1   1
10,4       0,1   1
10,8       0,1   1
10,32      0,1   1
10,256     0,1   1

# medium-4d
2,800,2,2     0,1,2,3   2
2,800,4,2     0,1,2,3   2
2,800,8,2     0,1,2,3   2
2,800,32,2    0,1,2,3   2
2,800,256,2   0,1,2,3   2

# small-4d
2,80,2,2     0,1,2,3   2
2,80,4,2     0,1,2,3   2
2,80,8,2     0,1,2,3   2
2,80,32,2    0,1,2,3   2
2,80,256,2   0,1,2,3   2
2,80,512,2   0,1,2,3   2

# medium-4d-trans
2,800,2,2     1,3,2,0   2
2,800,4,2     1,3,2,0   2
2,800,8,2     1,3,2,0   2
2,800,32,2    1,3,2,0   2
2,800,256,2   1,3,2,0   2

# small-4d-trans
2,80,2,2     1,3,2,0   2
2,80,4,2     1,3,2,0   2
2,80,8,2     1,3,2,0   2
2,80,32,2    1,3,2,0   2
2,80,256,2   1,3,2,0   2
2,80,512,2   1,3,2,0   2

# small-4d-contiguous
2,2,80,8     0,1,2,3   1
2,4,80,8     0,1,2,3   1
2,8,80,8     0,1,2,3   1
2,32,80,8    0,1,2,3   1
2,256,80,8   0,1,2,3   1
2,512,80,8   0,1,2,3   1

"""


# Some random permutations
SUITE += "# permutations\n"
random.seed(123)
for nd in xrange(2, 5):
    sizes = [2, 5, 7, 32, 200, 10000]
    transp = range(nd)
    for k in xrange(30 + 60/nd):
        random.shuffle(sizes)
        random.shuffle(transp)
        if reduce(lambda x, y: x*y, sizes[:nd])*8 > 200e6:
            continue
        SUITE += "%s  %s  %d\n" % (",".join(map(str, sizes[:nd])),
                                   ",".join(map(str, transp)),
                                   random.randint(0, nd-1))

# Completely random
SUITE += "# random\n"
random.seed(123)
for nd in xrange(2, 5):
    transp = range(nd)
    for k in xrange(30 + 60/nd):
        random.shuffle(transp)
        while True:
            sizes = [random.randint(2, 100) for x in xrange(nd)]
            if reduce(lambda x, y: x*y, sizes[:nd])*8 > 200e6:
                continue
            SUITE += "%s  %s  %d\n" % (",".join(map(str, sizes[:nd])),
                                       ",".join(map(str, transp)),
                                       random.randint(0, nd-1))
            break


def generate_suite():
    global SUITE
    items = {}
    section = "default"
    for x in SUITE.split("\n"):
        x = x.strip()
        if not x:
            continue
        if x.startswith('#'):
            section = x[1:].strip()
            continue
        items.setdefault(section, []).append(x)
    SUITE = items

generate_suite()

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

def main():
    alt_path = ""
    if os.path.isdir('numpy') and os.path.isdir('numpy'):
        os.chdir('build')
        try: alt_path = glob.glob('lib.*')[0]
        except: pass

    p = optparse.OptionParser(__doc__.strip())
    p.add_option("-t", "--time", action="store", type="float", default=1.0,
                 help="time for each benchmark (%default s)")
    p.add_option("-l", "--list", action="callback", callback=list_sections,
                 help="list available benchmarks")
    p.add_option("-f", "--fig-title", action="store", dest="title",
                 help="figure title", default="")
    options, args = p.parse_args()

    if len(args) == 1 and args[0] == 'plot':
        run_plot(options)
        return

    if len(args) == 0:
        p.error('no path given')

    if args[0] and not os.path.isdir(os.path.join(args[0], 'numpy')):
        p.error('path does not contain a numpy subdirectory')

    if len(args) == 1:
        run_suite(args[0], options)
    elif len(args) >= 2 and args[1][0] not in '0123456789':
        for sec in args[1:]:
            if sec not in SUITE:
                p.error('unknown test section "%s"' % sec)
        run_suite(args[0], options, sections=args[1:])
    else:
        try:
            new_path, shape, transpose, index = args
        except (ValueError, SyntaxError, IndexError):
            p.error('invalid arguments')

        run_single(new_path, shape, transpose, index, options)

def list_sections(option, opt, value, parser):
    print "Available benchmarks:"
    for sec in sorted(SUITE.keys()):
        print "-", sec
        for x in SUITE[sec]:
            print "    ", x
    raise SystemExit(0)


#------------------------------------------------------------------------------
# Plot
#------------------------------------------------------------------------------

def run_plot(options):
    import numpy as np
    import matplotlib.pyplot as plt

    items = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        x = line.split()
        items.append((x[0], x[1], x[2], x[3], tuple(map(float, x[4:]))))
        nsamples = len(x[4:])

    dt = [('code', 'S3'), ('shape', 'S64'), ('trans', 'S64'),
          ('axis', int), ('res', float, (nsamples,))]

    d = np.array(items, dtype=dt)
    x = np.zeros(d.shape)
    y = np.zeros(d.shape + (nsamples,))

    for j, m in enumerate(d):
        shape = map(int, m['shape'].split(','))
        trans = map(int, m['trans'].split(','))
        shape = np.array(shape)[trans]
        axis = m['axis']

        n_elements = np.prod(shape)

        x[j] = n_elements
        y[j] = m['res'] * n_elements

    # jitter positions
    counts = {}
    nsame = {}
    for j, m in enumerate(d):
        xx = float(x[j])
        counts[xx] = counts.get(xx, 0) + 1
        nsame.setdefault(xx, (x == xx).sum())
        x[j] += (counts[xx] * 1.0 / (nsame[xx] - 1) - .5) * 0.05 * xx

    j_new = (d['code'] == 'new')
    j_old = (d['code'] == 'old')

    xx = np.c_[x[j_new], x[j_old]]
    yy = np.c_[y[j_new].max(axis=1), y[j_old].max(axis=1)]

    j_imp = (yy[:,0] > yy[:,1]*1.25)
    j_reg = (yy[:,0] < yy[:,1]/1.25)
    j_neu = (~j_imp & ~j_reg)

    print "Regressions:"
    print d[np.where(j_new)[0][j_reg]], "\n"

    print "Geometric mean:"
    print "new:", np.exp(np.log(y[j_new]).mean())
    print "old:", np.exp(np.log(y[j_old]).mean())

    plt.rc('figure', figsize=(8, 4.94427))

    ds_imp = plt.loglog(xx[j_imp].T, yy[j_imp].T, 'b-', zorder=10)
    ds_neu = plt.loglog(xx[j_neu].T, yy[j_neu].T, 'k-', lw=2, zorder=10)
    ds_reg = plt.loglog(xx[j_reg].T, yy[j_reg].T, 'r-', lw=2, zorder=10)

    #ls1 = plt.loglog(x[j_old], y[j_old].max(axis=1), 'cs', markersize=4.0,
    #                 zorder=10)
    #ls2 = plt.loglog(x[j_new], y[j_new].max(axis=1), 'm.', markersize=3.0,
    #                 zorder=10)
    plt.xlabel(r'elements')
    plt.ylabel(r'ops / s')
    plt.gca().yaxis.grid(1, which='major', color=(0.3, 0.3, 0.3), zorder=-1)
    plt.gca().yaxis.grid(1, which='minor', color=(0.8, 0.8, 0.8), zorder=-1)
    plt.ylim(1e6, 6e8)
    #plt.legend((ls1[0], ls2[0]), ('old', 'new'), loc='best')
    plt.legend((ds_imp[0], ds_neu[0], ds_reg[0]),
               ('improvement', 'neutral', 'regression'), loc='best')

    for q in [64, 512, 6144]:
        plt.axvline(q*1024/8, color='k', ls=':')
        plt.text(q*1024/8*1.2, plt.ylim()[0]*1.2,
                 '%d kb' % q, ha='left', va='bottom')

    if options.title:
        plt.title(options.title)

    plt.savefig('test.png')
    plt.show()


#------------------------------------------------------------------------------
# Run timers
#------------------------------------------------------------------------------

def run_suite(new_path, options, sections=None):
    fn = os.path.abspath(__file__)
    opts = ['-t', repr(options.time)]

    for sec in sorted(SUITE.keys()):
        if sections is not None and sec not in sections:
            continue
        print "#", sec
        sys.stdout.flush()
        for item in SUITE[sec]:
            subprocess.call([sys.executable, fn, new_path]+ opts + item.split())
            subprocess.call([sys.executable, fn, ''] + opts + item.split())

def run_single(new_path, shape, transpose, index, options):
    if new_path:
        sys.path.insert(0, new_path)
        new = 'new'
    else:
        new = 'old'

    import numpy

    pre_code = """
import numpy as np
#np.setbufsize(512*1024/8)

s = np.random.randn(%s)

cache_size = 10000 * 1024 / 8
ss = []
for j in xrange(5 + 15*cache_size//s.size):
    ss.append(s.copy().transpose(%s))

jx = [0]
def get():
    jx[0] = (jx[0] + 1) %% len(ss)
    return ss[jx[0]]

""" % (shape, transpose)

    code = "np.sum(get(), axis=%s)" % index

    ns = {}
    exec pre_code in ns

    ts = magic_timeit(code, ns=ns, secs=options.time, repeat=5)

    print new, shape, transpose, index, "  ".join([
        "%.3g" % (1/t) for t in ts])


#------------------------------------------------------------------------------
# Timing
#------------------------------------------------------------------------------

def magic_timeit(stmt, ns=None, secs=4.0, repeat=timeit.default_repeat):
    timefunc = timeit.default_timer
    number = 0
    timer = timeit.Timer(timer=timefunc)

    src = timeit.template % {'stmt': timeit.reindent(stmt, 8),
                             'setup': "pass"}
    code = compile(src, "<magic-timeit>", "exec")

    if ns is None:
        ns = {}
    exec code in ns
    timer.inner = ns["inner"]

    if number == 0:
        # determine number so that 0.4 <= total time < 4.0
        number = 4
        done = 0
        timed = 0
        for i in range(1, 10):
            number *= 2
            timed += timer.timeit(number)
            done += number
            if timed >= 0.05:
                break

        number = 1 + int(done * secs / timed / repeat)

    a = time.time()
    res = [x / number for x in timer.repeat(repeat, number)]

    return res

#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
