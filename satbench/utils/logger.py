import sys


class Logger(object):
    def __init__(self, filename=None, stream=sys.stdout):
        if filename is not None:
            self.log = open(filename, 'a')
        else:
            self.log = None
        self.terminal = stream

    def write(self, message):
        self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
    
    def flush(self):
        pass
