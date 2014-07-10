import sys

class TrainingFailedError(RuntimeError):
    """
    Raised indicicating the Training Failed, 
    and the model is in a invalid state to read output"""
    def __init__(self, inner_error):
        self.inner_error = inner_error
        self.traceback = sys.exc_info()
        RuntimeError.__init__(self, "Training Error: " + inner_error.message)

