#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function

from io import StringIO as UStringIO
import sys, os
import re

def cleanSource(source):
    # remove whitespace at end of lines
    source = re.sub(r'[\t ]+\n', '\n', source)
    # remove comments like .. Scalar Arguments ..
    source = re.sub(r'(?m)^[\t ]*/\* *\.\. .*?\n', '', source)
    # collapse blanks of more than two in-a-row to two
    source = re.sub(r'\n\n\n\n+', r'\n\n\n', source)
    return source

class LineQueue:
    def __init__(self):
        object.__init__(self)
        self._queue = []

    def add(self, line):
        self._queue.append(line)

    def clear(self):
        self._queue = []

    def flushTo(self, other_queue):
        for line in self._queue:
            other_queue.add(line)
        self.clear()

    def getValue(self):
        q = LineQueue()
        self.flushTo(q)
        s = ''.join(q._queue)
        self.clear()
        return s

class CommentQueue(LineQueue):
    def __init__(self):
        LineQueue.__init__(self)

    def add(self, line):
        if line.strip() == '':
            LineQueue.add(self, '\n')
        else:
            line = '  ' + line[2:-3].rstrip() + '\n'
            LineQueue.add(self, line)

    def flushTo(self, other_queue):
        if len(self._queue) == 0:
            pass
        elif len(self._queue) == 1:
            other_queue.add('/*' + self._queue[0][2:].rstrip() + ' */\n')
        else:
            other_queue.add('/*\n')
            LineQueue.flushTo(self, other_queue)
            other_queue.add('*/\n')
        self.clear()

# This really seems to be about 4x longer than it needs to be
def cleanComments(source):
    lines = LineQueue()
    comments = CommentQueue()
    def isCommentLine(line):
        return line.startswith('/*') and line.endswith('*/\n')

    blanks = LineQueue()
    def isBlank(line):
        return line.strip() == ''

    def SourceLines(line):
        if isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            lines.add(line)
            return SourceLines
    def HaveCommentLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            lines.add(line)
            return SourceLines
    def HaveBlankLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            blanks.flushTo(comments)
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            blanks.flushTo(lines)
            lines.add(line)
            return SourceLines

    state = SourceLines
    for line in UStringIO(source):
        state = state(line)
    comments.flushTo(lines)
    return lines.getValue()

def removeHeader(source):
    lines = LineQueue()

    def LookingForHeader(line):
        m = re.match(r'/\*[^\n]*-- translated', line)
        if m:
            return InHeader
        else:
            lines.add(line)
            return LookingForHeader
    def InHeader(line):
        if line.startswith('*/'):
            return OutOfHeader
        else:
            return InHeader
    def OutOfHeader(line):
        if line.startswith('#include "f2c.h"'):
            pass
        else:
            lines.add(line)
        return OutOfHeader

    state = LookingForHeader
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def removeSubroutinePrototypes(source):
    expression = re.compile(
        r'/[*] Subroutine [*]/^\s*(?:(?:inline|static)\s+){0,2}(?!else|typedef|return)\w+\s+\*?\s*(\w+)\s*\([^0]+\)\s*;?'
    )
    lines = LineQueue()
    for line in UStringIO(source):
        if not expression.match(line):
            lines.add(line)

    return lines.getValue()

def removeBuiltinFunctions(source):
    lines = LineQueue()
    def LookingForBuiltinFunctions(line):
        if line.strip() == '/* Builtin functions */':
            return InBuiltInFunctions
        else:
            lines.add(line)
            return LookingForBuiltinFunctions

    def InBuiltInFunctions(line):
        if line.strip() == '':
            return LookingForBuiltinFunctions
        else:
            return InBuiltInFunctions

    state = LookingForBuiltinFunctions
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def replaceDlamch(source):
    """Replace dlamch_ calls with appropriate macros"""
    def repl(m):
        s = m.group(1)
        return dict(E='EPSILON', P='PRECISION', S='SAFEMINIMUM',
                    B='BASE')[s[0]]
    source = re.sub(r'dlamch_\("(.*?)", \(ftnlen\)[0-9]+\)', repl, source)
    source = re.sub(r'^\s+extern.*? dlamch_.*?;$(?m)', '', source)
    return source

# do it

def scrubSource(source, nsteps=None, verbose=False):
    steps = [
             ('remove header', removeHeader),
             ('clean source', cleanSource),
             ('clean comments', cleanComments),
             ('replace dlamch_() calls', replaceDlamch),
             ('remove prototypes', removeSubroutinePrototypes),
             ('remove builtin function prototypes', removeBuiltinFunctions),
            ]

    if nsteps is not None:
        steps = steps[:nsteps]

    for msg, step in steps:
        if verbose:
            print(msg)
        source = step(source)

    return source

if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = os.path.join(sys.argv[2], os.path.basename(filename))
    with open(filename, 'r') as fo:
        source = fo.read()

    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    else:
        nsteps = None

    source = scrub_source(source, nsteps, verbose=True)

    writefo = open(outfilename, 'w')
    writefo.write(source)
    writefo.close()
