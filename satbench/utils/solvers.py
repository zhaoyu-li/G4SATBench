import os
import re
import signal
import time
import subprocess
import numpy as np
import pickle

from decimal import Decimal


class SATSolver:
    def __init__(self, opts):
        self.opts = opts
        if self.opts.solver == 'Glucose4':
            self.exec_dir = os.path.abspath('satbenchmark/external/glucose')
            self.cmd_line = ['./glucose']
            if self.opts.model:
                self.cmd_line.append('-model')
            if self.opts.proof:
                self.cmd_line = self.cmd_line + ['-vbyte', '-certified']

    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        cmd_line = self.cmd_line.copy()

        if self.opts.proof:
            tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '.out')
            cmd_line.append('-certified-output')
            cmd_line.append(input_filepath)

        with open(tmp_filepath, 'w') as f:
            t0 = time.time()
            timeout_expired = 0
            try:
                process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate(timeout=self.opts.timeout)
                # may also finished by linux oom killer
            except:
                timeout_expired = 1
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            t = time.time() - t0
        
        complete = 0
        assignment = None
        num_flips = 0

        if timeout_expired or os.stat(tmp_filepath).st_size == 0: # timeout
            os.remove(tmp_filepath)
            return complete, assignment, num_flips, t
        
        with open(tmp_filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('v'):
                    assignment = np.array([int(s) for s in line.strip().split()[1:-1]]) > 0
                if line.startswith('c numFlips'): # Local search solver
                    num_flips = Decimal(line.strip().split()[-1])
        
        if assignment is not None: # All instances are SAT
            complete = 1
        
        os.remove(tmp_filepath)
        return complete, assignment, num_flips, t
